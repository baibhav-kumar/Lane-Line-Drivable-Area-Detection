import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *

LOGGING_NAME = "custom"

def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})

set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def netParams(model):
    """Calculate total number of parameters"""
    return sum([param.nelement() for param in model.parameters()])

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    torch.save(state, filename)

def train(args, train_loader, model, criterion, optimizer, epoch):
    """Training function"""
    model.train()
    
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    losses = AverageMeter()
    focal_losses = AverageMeter()
    tversky_losses = AverageMeter()
    
    total_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=total_batches, 
                desc=f'Epoch {epoch:3d}/{args.max_epochs-1:3d}', 
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for i, (image_name, input, target) in pbar:
        if args.onGPU == True:
            input = input.cuda().float() / 255.0
        
        target_da, target_ll = target
        if args.onGPU == True:
            target_da = target_da.cuda()
            target_ll = target_ll.cuda()
        
        output = model(input)
        focal_loss, tversky_loss, loss = criterion(output, (target_da, target_ll))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), input.size(0))
        focal_losses.update(focal_loss.item(), input.size(0))
        tversky_losses.update(tversky_loss.item(), input.size(0))
        
        pbar.set_description(f'Epoch {epoch:3d}/{args.max_epochs-1:3d} | '
                           f'Focal: {focal_losses.avg:.4f} | '
                           f'Tversky: {tversky_losses.avg:.4f} | '
                           f'Total: {losses.avg:.4f}')
    
    return losses.avg

@torch.no_grad()
def val(val_loader, model):
    """Standard validation without TTA"""
    model.eval()
    
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    
    total_batches = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=total_batches, desc='Validating...')
    
    for i, (image_name, input, target) in pbar:
        input = input.cuda().float() / 255.0
        target_da, target_ll = target
        target_da = target_da.cuda()
        target_ll = target_ll.cuda()
        
        output = model(input)
        out_da, out_ll = output
        
        _, da_predict = torch.max(out_da, 1)
        _, da_gt = torch.max(target_da, 1)
        _, ll_predict = torch.max(out_ll, 1)
        _, ll_gt = torch.max(target_ll, 1)
        
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())
        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()
        
        da_acc_seg.update(da_acc, input.size(0))
        da_IoU_seg.update(da_IoU, input.size(0))
        da_mIoU_seg.update(da_mIoU, input.size(0))
        
        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())
        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()
        
        ll_acc_seg.update(ll_acc, input.size(0))
        ll_IoU_seg.update(ll_IoU, input.size(0))
        ll_mIoU_seg.update(ll_mIoU, input.size(0))
    
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    
    return da_segment_result, ll_segment_result


@torch.no_grad()
def val_tta(val_loader, model, use_flip=True):
    """
    Validation with Test-Time Augmentation (TTA)
    
    Args:
        val_loader: validation data loader
        model: trained model
        use_flip: if True, apply horizontal flip augmentation
    
    Returns:
        Tuple of (da_results, ll_results) where each is (acc, IoU, mIoU)
    """
    model.eval()
    
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    
    total_batches = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=total_batches, desc='TTA Validation')
    
    for i, (image_name, input, target) in pbar:
        input = input.cuda().float() / 255.0
        target_da, target_ll = target
        target_da = target_da.cuda()
        target_ll = target_ll.cuda()
        
        # Original prediction
        output = model(input)
        out_da, out_ll = output
        
        # TTA: Horizontal flip
        if use_flip:
            input_flip = torch.flip(input, [3])  # Flip width dimension
            output_flip = model(input_flip)
            out_da_flip, out_ll_flip = output_flip
            
            # Flip predictions back
            out_da_flip = torch.flip(out_da_flip, [3])
            out_ll_flip = torch.flip(out_ll_flip, [3])
            
            # Average predictions
            out_da = (out_da + out_da_flip) / 2
            out_ll = (out_ll + out_ll_flip) / 2
        
        # Get predictions
        _, da_predict = torch.max(out_da, 1)
        _, da_gt = torch.max(target_da, 1)
        _, ll_predict = torch.max(out_ll, 1)
        _, ll_gt = torch.max(target_ll, 1)
        
        # Drivable Area metrics
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())
        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()
        
        da_acc_seg.update(da_acc, input.size(0))
        da_IoU_seg.update(da_IoU, input.size(0))
        da_mIoU_seg.update(da_mIoU, input.size(0))
        
        # Lane Line metrics
        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())
        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()
        
        ll_acc_seg.update(ll_acc, input.size(0))
        ll_IoU_seg.update(ll_IoU, input.size(0))
        ll_mIoU_seg.update(ll_mIoU, input.size(0))
        
        pbar.set_description(f'TTA Val | DA mIoU: {da_mIoU_seg.avg:.4f} | LL IoU: {ll_IoU_seg.avg:.4f}')
    
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    
    return da_segment_result, ll_segment_result
