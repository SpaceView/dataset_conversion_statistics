"""
INTRODUCTION: 
    4096x3000 ----> CropImage to 3000x3000 ----> ResizeImage to 800x800
    the target of crop is to ensure that all the resizing is in equal proportion, 
    in both X and Y dimension
NOTE: 
you can set the below constants to obtain different image sizes:
    CROP_WIDTH, 
    CROP_HEIGHT
    TARGET_WIDTH
    TARGET_HEIGHT
"""

import json
import numpy as np
import copy

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from collections import defaultdict
import sys

import os
from os import listdir
from os.path import isfile, join
import pathlib
import shutil

import cv2

#NOTE: the original image dimension is 4096x3000
#      the image is in the center in ROUND shape
#THUS: we crop this image to 3000x3000
CROP_WIDTH = 3000
CROP_HEIGHT = 3000
crop_req = True

#NOTE: this is the annotation image dimension
TARGET_WIDTH = 800
TARGET_HEIGHT = 800
resize_req = True

#NOTE: 
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)

img_org_root = 'E:/EsightData/JX05/'
img_tgt_root = 'E:/EsightData/JX05PNG/'

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = img_org_root
os.environ["TARGET_PATH"] = img_tgt_root

data_root = os.environ['IMG_ROOT_PATH']
target_root = os.environ['TARGET_PATH']

def run_fast_scandir(dir, ext):    #dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files

def CropImageToPredefinedSize(_src):
    height, width, ch = _src.shape
    assert(width >= CROP_WIDTH)
    assert(height >= CROP_HEIGHT)
    x0 = int((width - CROP_WIDTH)/2)
    y0 =int((height - CROP_HEIGHT)/2)
    x1 = x0 + CROP_WIDTH
    y1 = y0 + CROP_HEIGHT
    dst = _src[y0:y1, x0:x1]
    return dst

if __name__ == "__main__":
    subfolders, files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])

    for fld in subfolders:
        print(fld)

    dir = os.path.dirname(img_tgt_root)
    if not (os.path.exists(dir)):
        os.makedirs(dir)

    print("----------------------------------------->Start Crop and Resize<-----------------------------------------")
    for f in files:
        print(f)        
        frp = f.replace(img_org_root, img_tgt_root)
        sfx = pathlib.Path(f).suffix
        ftgt = frp.replace(sfx, '.png')        
        print(ftgt)

        img_org = cv2.imread(f)
        if(img_org is None):
            continue

        img_crop = CropImageToPredefinedSize(_src=img_org)
        img_height, img_width, img_ch = img_crop.shape
        assert(CROP_HEIGHT == img_height)
        assert(CROP_WIDTH == img_width)
        assert(3 == img_ch)

        t_dim = (TARGET_WIDTH, TARGET_HEIGHT)
        img_rsz = cv2.resize(src=img_crop, dsize=t_dim, interpolation=cv2.INTER_AREA)
        img_height, img_width, img_ch = img_rsz.shape
        assert(TARGET_HEIGHT == img_height)
        assert(TARGET_WIDTH == img_width)
        assert(3 == img_ch)
        img_tgt_dir = os.path.dirname(ftgt)

        if(not os.path.exists(img_tgt_dir)):
            os.makedirs(img_tgt_dir)   #NOTE: os.mkdir will not make intermediate folders

        cv2.imwrite(ftgt, img_rsz)

    print('Main Done!')

print('All done!')


