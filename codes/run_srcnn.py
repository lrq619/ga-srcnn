import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
import time
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

def run_srcnn(dir):
    file_count = len(os.listdir(dir))
    srcnn_dir = dir.split('_')[0]+'_srcnn'
    print(srcnn_dir)
    if not os.path.exists(srcnn_dir):
        os.mkdir(srcnn_dir)

    weights_file = 'models/best_60.pth'
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    sum_time = 0
    for file in os.listdir(dir):
        image_file = '%s/%s'%(dir,file)
        print('\rSRCNN %s'%image_file,end='',flush=True)
        image = pil_image.open(image_file).convert('RGB')

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)
        end = time.time()
        sum_time += (end - start) * 1000
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save('%s/%s'%(srcnn_dir,file))
    return (sum_time / file_count)
