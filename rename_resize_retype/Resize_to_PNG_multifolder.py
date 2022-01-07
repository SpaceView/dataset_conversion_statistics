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

import cv2

TARGET_WIDTH = 800
TARGET_HEIGHT = 600
crop_req = True
resize_req = True
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)

img_org_root = 'E:/ESight/GoldLineOrg/'
img_tgt_root = 'E:/ESight/GoldLine/'


#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = img_org_root
os.environ["TARGET_PATH"] = img_tgt_root


data_root = os.environ['IMG_ROOT_PATH']
target_root = os.environ['TARGET_PATH']


def run_fast_scandir(dir, ext):    # dir: str, ext: list
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


"""
#NOTE: if you don't need to list folders recursively, use the below code
onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
image_root = os.path.join(data_root, 'subsoure')
target_root = os.path.join(data_root, 'subtarget')
if not os.path.exists(target_root):
    os.makedirs(target_root)
"""

def resize_image(img_in):
    ht, wd = img_in.shape[0:2]
    ry = TARGET_HEIGHT / ht
    rx = TARGET_WIDTH /wd
    scale = 0
    if(ry>rx):
        scale = ry
    else:
        scale = rx
    dest_ht = int(ht * scale)
    dest_wd = int(wd * scale)
    dim = (dest_wd, dest_ht)
    img_resized = cv2.resize(img_in, dim, interpolation=cv2.INTER_AREA)
    return img_resized


if __name__ == "__main__":
    subfolders, files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])

    for fld in subfolders:
        print(fld)

    for f in files:
        print(f)
        ft = f.replace(img_org_root, img_tgt_root)
        fpng = ft.rsplit( ".", 1 )[0] + '.png'
        print(fpng)

        dir = os.path.dirname(fpng)
        if not (os.path.exists(dir)):
            os.makedirs(dir)

        img = cv2.imread(f)
        if(crop_req):
            img = img[0:3000, 0:4000, :]
        if(resize_req):
            img = resize_image(img)

        cv2.imwrite(fpng, img)

    print('Main Done!')

print('All done!')

