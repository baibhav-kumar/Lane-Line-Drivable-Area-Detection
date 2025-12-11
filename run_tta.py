import torch
from model.TwinLite import TwinLiteNet
import DataSet as myDataLoader
from utils import val, val_tta

# Load model
model = TwinLiteNet()

# Load weights BEFORE wrapping in DataParallel
state_dict = torch.load('exp_ll5/best_model.pth')
model.load_state_dict(state_dict)

# NOW wrap in DataParallel and move to GPU
model = torch.nn.DataParallel(model).cuda()
model.eval()

valLoader = torch.utils.data.DataLoader(
    myDataLoader.MyDataset(root_dir='/home/iotlab/iotlab_project/TwinLiteNet/data/bdd100k', valid=True),
    batch_size=4, shuffle=False, num_workers=4, pin_memory=True
)

print("\n" + "="*70)
print("BASELINE (No TTA)")
print("="*70)
da_base, ll_base = val(valLoader, model)
print(f"DA mIoU: {da_base[2]*100:.2f}% | LL IoU: {ll_base[1]*100:.2f}%")

print("\n" + "="*70)
print("WITH TTA (Horizontal Flip)")
print("="*70)
da_tta, ll_tta = val_tta(valLoader, model, use_flip=True)
print(f"DA mIoU: {da_tta[2]*100:.2f}% | LL IoU: {ll_tta[1]*100:.2f}%")

print("\n" + "="*70)
print(f"IMPROVEMENT: DA +{(da_tta[2]-da_base[2])*100:.2f}% | LL +{(ll_tta[1]-ll_base[1])*100:.2f}%")
print("="*70)
