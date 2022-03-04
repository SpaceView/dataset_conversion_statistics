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

TARGET_WIDTH = 800
TARGET_HEIGHT = 600
crop_req = True
resize_req = True
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)

img_org_root = 'E:/EsightData/0218test/original'
img_tgt_root = 'E:/EsightData/0218test/rename'
img_tgt_dir  = img_tgt_root + '/'

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
#os.environ["IMG_ROOT_PATH"] = img_org_root
#os.environ["TARGET_PATH"] = img_tgt_root
#data_root = os.environ['IMG_ROOT_PATH']
#target_root = os.environ['TARGET_PATH']


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


if __name__ == "__main__":
    subfolders, files = run_fast_scandir(img_org_root, [".bmp", ".png", ".jpg", ".jpeg"])

    for fld in subfolders:
        print(fld)

    dir = img_tgt_dir #os.path.dirname(img_tgt_root)
    if not (os.path.exists(dir)):
        os.makedirs(dir)

    for f in files:
        print(f)
        fj_src = f.rsplit( ".", 1 )[0] + '.json'

        fstem = f.replace(img_org_root, "img_tgt_root_placeholder_")
        fstem = fstem.rsplit( ".", 1 )[0]
        #fpt = fpt.rsplit( ".", 1 )[0] + '.png'
        fpt_list = fstem.split( "\\")
        fpt = ""
        for fl in fpt_list:
            fpt = fpt + "_" + fl
        fpt = fpt.replace("_img_tgt_root_placeholder__", img_tgt_dir)
        fj_tgt = fpt + '.json'
        fp_tgt = fpt + '.png'
        print(fp_tgt)

        img = cv2.imread(f)
        cv2.imwrite(fp_tgt, img)
        
        if(os.path.exists(fj_src)):
            shutil.copy(fj_src, fj_tgt)        

    print('Main Done!')

print('All done!')

