import os
import torch
import pickle
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from loss import TotalLoss

def train_net(args):
    # Load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    
    print(f"CUDA Available: {cuda_available}")
    print(f"Number of GPUs: {num_gpus}")
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    model = net.TwinLiteNet()
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    args.savedir = args.savedir + '/'
    
    # Create directory if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    # DataLoader with root_dir parameter
    print(f"\nLoading dataset from: {args.data_root}")
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(root_dir=args.data_root, valid=False),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(root_dir=args.data_root, valid=True),
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
        print("Model moved to GPU")
    else:
        args.onGPU = False
        print("WARNING: Running on CPU - training will be very slow!")
    
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    # Enhanced loss with weighted lane line loss
    criteria = TotalLoss(ll_weight=args.ll_weight)
    
    start_epoch = 0
    lr = args.lr
    
    # Use AdamW for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr, 
        betas=(0.9, 0.999), 
        eps=1e-08, 
        weight_decay=args.weight_decay
    )
    
    # Add cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.max_epochs, 
        eta_min=1e-6
    )
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    best_miou = 0.0
    best_ll_iou = 0.0
    
    print("\nStarting training...\n")
    
    for epoch in range(start_epoch, args.max_epochs):
        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        
        # Get current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("\n" + "="*70)
        print(f"Epoch [{epoch}/{args.max_epochs}] Learning rate: {lr:.6f}")
        print("="*70)
        
        # Train for one epoch
        model.train()
        train(args, trainLoader, model, criteria, optimizer, epoch)
        
        # Validation (val function doesn't take args in original utils.py)
        model.eval()
        da_segment_results, ll_segment_results = val(valLoader, model)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Driving Area  - Acc: {da_segment_results[0]:.4f}, IoU: {da_segment_results[1]:.4f}, mIoU: {da_segment_results[2]:.4f}")
        print(f"  Lane Line     - Acc: {ll_segment_results[0]:.4f}, IoU: {ll_segment_results[1]:.4f}, mIoU: {ll_segment_results[2]:.4f}")
        
        # Save best model
        current_miou = da_segment_results[2]
        current_ll_iou = ll_segment_results[1]
        
        if current_miou > best_miou or current_ll_iou > best_ll_iou:
            best_miou = max(best_miou, current_miou)
            best_ll_iou = max(best_ll_iou, current_ll_iou)
            torch.save(model.state_dict(), args.savedir + 'best_model.pth')
            print(f"\n*** NEW BEST MODEL! DA mIoU: {best_miou:.4f}, LL IoU: {best_ll_iou:.4f} ***\n")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_file_name)
            print(f"Saved checkpoint: {model_file_name}")
        
        # Save training checkpoint with all info
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'lr': lr,
            'best_miou': best_miou,
            'best_ll_iou': best_ll_iou
        }, args.savedir + 'checkpoint.pth.tar')
        
        # Step scheduler
        scheduler.step()
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best DA mIoU: {best_miou:.4f}")
    print(f"Best LL IoU: {best_ll_iou:.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/ceec/huycq/data/bdd100k', 
                        help='Root directory of BDD100K dataset')
    parser.add_argument('--max_epochs', type=int, default=150, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--ll_weight', type=float, default=3.5, help='Lane line loss weight for class imbalance')
    parser.add_argument('--savedir', default='./exp_enhanced', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--onGPU', default=True, type=bool, help='Train on GPU')
    
    args = parser.parse_args()
    train_net(args)
