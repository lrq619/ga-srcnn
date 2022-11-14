from math import log10, sqrt
import cv2
from run_srcnn import run_srcnn
import numpy as np
import torchvision
from config import GAME
import os
import csv

def calc_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore calc_psnr have no importance.
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def test_psnr(dir1,dir2):
    file_count = len(os.listdir(dir1))
    if file_count != len(os.listdir(dir2)):
        print("dir1 and dir2 must have same number of image files")
        return -1
    sum_psnr = 0
    for file in os.listdir(dir1):
        img1 = cv2.imread('%s/%s'%(dir1,file))
        img2_file = '%s/%s'%(dir2,file)
        if not os.path.exists(img2_file):
            print('Unmatched file for %s in %s'%(file,dir2))
            return -1
        img2 = cv2.imread(img2_file)
        sum_psnr += calc_psnr(img1,img2)
    return (sum_psnr/file_count)
    

clear_dir = '%s/clear'%GAME
clear_low_dir = '%s/clear_low'%GAME
blur_dir = '%s/blur'%GAME
blur_low_dir = '%s/blur_low'%GAME
clear_srcnn_dir = '%s/clear_srcnn'%GAME

srcnn_time = run_srcnn(clear_low_dir)

clear_low_psnr = test_psnr(clear_dir,clear_low_dir)
clear_srcnn_psnr = test_psnr(clear_dir,clear_srcnn_dir)

blur_low_psnr = test_psnr(blur_dir,blur_low_dir)

in_file = open('%s/%s.csv'%(GAME,GAME), 'r')
reader = csv.reader(in_file)
table = list(reader)
in_file.close()

out_file = open('%s/%s.csv'%(GAME,GAME), "w",newline='')
writer = csv.writer(out_file)

table[1][2] = clear_low_psnr
table[2][2] = blur_low_psnr
writer.writerows(table)
writer.writerow(['(Clear,SRCNN)','%.5fms'%srcnn_time,clear_srcnn_psnr])
out_file.close()
    

