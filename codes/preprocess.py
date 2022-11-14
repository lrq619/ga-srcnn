import os
from torchvision import transforms
import PIL.Image as pil_image
import time
import csv
from config import GAME, DOWNMAG, NEEDBLUR
import numpy as np
import cv2

NEEDBLUR = 1

def rename(dir):
    count = 1
    for file in os.listdir(dir):
        try:
            os.rename('%s/%s'%(dir,file), '%s/%d.png'%(dir,count))
            count += 1
        except:
            pass


        

def reduce_resolution(img,downmag):
    w = img.width
    h = img.height
    down_size = (int(h/downmag),int(w/downmag))
    
    down = transforms.Resize(down_size)
    start = time.time()
    up = transforms.Resize((h,w))
    end = time.time()
    res = down(img)
    res = up(res)
    return res,end-start




def create_blur_imgs(dir):
    # dir_name = dir.split('/')[-1]
    dir_blur_name = 'blur'
    blur_dir = '%s/%s'%(GAME,dir_blur_name)
    
    if not os.path.exists(blur_dir):
        os.mkdir(blur_dir)

    kernel_size = 30

        # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    for file in os.listdir(dir):
        print('\rBlurring %s'%file,end='',flush=True)
        img = cv2.imread('%s/%s'%(dir,file))
        
        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(img, -1, kernel_v)
        cv2.imwrite('%s/%s'%(blur_dir,file),vertical_mb)
        
        
        
    

def create_low_imgs(dir):
    count = 0
    dir_name = dir.split('/')[-1]
    dir_low_name = dir_name+'_low'
    low_dir = './%s/%s'%(GAME,dir_low_name)
    total_dura = 0
    if not os.path.exists(low_dir):
        os.mkdir(low_dir)
    for file in os.listdir(dir):
        print('\rProcessing %s'%file,end='',flush=True)
        orig_img = pil_image.open('%s/%s'%(dir,file)).convert('RGB')
        low_img,duration = reduce_resolution(orig_img, DOWNMAG)
        low_img.save('%s/%s'%(low_dir,file))
        total_dura += duration
        count += 1
    return total_dura/count
        

clear_dir = './%s/clear'%GAME
blur_dir =  './%s/blur'%GAME

if NEEDBLUR:
    create_blur_imgs(clear_dir)

clear_count = len(os.listdir(clear_dir))
blur_count = len(os.listdir(blur_dir))

rename(clear_dir)
rename(blur_dir)

clear_upsample_time = create_low_imgs(clear_dir)
#upsample using srcnn
blur_upsample_time = create_low_imgs(blur_dir)
with open('%s/%s.csv'%(GAME,GAME),'w+',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['','Latency','PSNR'])
    writer.writerow(['(Clear,Bicubic)','%.5f ms'%(clear_upsample_time*1000),'0'])
    writer.writerow(['(Blur,Bicubic)','%.5f ms'%(blur_upsample_time*1000),'0'])

