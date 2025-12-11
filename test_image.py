import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import cv2
from model import TwinLite as net

def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    
    with torch.no_grad():
        img_out = model(img)
    
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    
    img_rs[DA > 100] = [255, 0, 0]  # Red for drivable area
    img_rs[LL > 100] = [0, 255, 0]  # Green for lane lines
    
    return img_rs


if __name__ == '__main__':
    print("Loading model...")
    model = net.TwinLiteNet()
    
    # FIXED: Load weights BEFORE DataParallel
    model.load_state_dict(torch.load('exp_ll5/best_model.pth'))
    
    # THEN wrap with DataParallel and move to GPU
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create output directory
    if os.path.exists('Final Results'):
        shutil.rmtree('Final Results')
    os.mkdir('newresults')
    
    # Process images
    image_list = os.listdir('images')
    print(f"\nProcessing {len(image_list)} images...")
    
    for i, imgName in enumerate(tqdm(image_list, desc="Processing")):
        img = cv2.imread(os.path.join('images', imgName))
        if img is None:
            print(f"Warning: Could not read {imgName}")
            continue
        
        img_result = Run(model, img)
        cv2.imwrite(os.path.join('Final Results', imgName), img_result)
    
    print(f"\n✓ Done! Results saved to 'newresults/' folder")
