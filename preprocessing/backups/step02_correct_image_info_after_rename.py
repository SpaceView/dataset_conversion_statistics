import json
import numpy as np
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from collections import defaultdict

import os
from os import listdir
from os.path import isfile, join
import pathlib

import cv2
import base64

import math
from itertools import groupby

import sys
#sys.path.insert(1, 'D:/py/dataset_conversion_statistics/')  
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

from img_utils.mask_to_bbox import mask_to_bbox

#CROP_WIDTH = 4000
#CROP_HEIGHT = 3000
#TARGET_WIDTH = 600
#TARGET_HEIGHT = 450  #600*600/800
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)
PI = 3.1415926535897932
MASKCHAR = 255
#crop_req = False
#resize_req = True

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/0221/final/img/rename_org'
os.environ["TGT_ROOT_PATH"] = 'E:/EsightData/0221/final/img/renameTgt'

data_root = os.environ['IMG_ROOT_PATH']
image_root = data_root
ann_root = data_root            #os.path.join(data_root, 'coco')

tgt_root = os.environ["TGT_ROOT_PATH"]
if not (os.path.exists(tgt_root)):
        os.makedirs(tgt_root)

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_imgPaths = []
my_jsonfiles = []
my_jsonPaths = []

for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
        label_file = pathlib.Path(onlyfiles[i]).stem + '.txt'
        annotation_file = os.path.join(ann_root, json_file)
        img_file = os.path.join(image_root, onlyfiles[i])
        if not isfile(annotation_file):
            #os.remove(img_file)
            print("--------> empty image file (no corresponding annotations): ", img_file)
            empty_images_count = empty_images_count + 1
            continue
        my_imgfiles.append(onlyfiles[i])
        my_imgPaths.append(img_file)
        my_jsonfiles.append(json_file)        
        my_jsonPaths.append(annotation_file)

#===========================================================================

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)                
    if not (0 == len(subfolders)):
        for dir in list(subfolders):
            sf, f = run_fast_scandir(dir, ext)
            subfolders.extend(sf)
            files.extend(f)
    return subfolders, files

def rleToMask(rleNumbers,height,width):
    #NOTE: rleNumbers = [67348, 6, 592, 9, 590, 11, 588, 13, 586, 15, 585, 16, 583, 17, ...]
    rows,cols = height,width    
    #rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rleLen = len(rleNumbers)
    if(len(rleNumbers) % 2):
        rleLen = rleLen - 1
        rleNumbers = rleNumbers[:rleLen]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    msk = np.zeros(rows*cols,dtype=np.uint8)
    tot_start = 0
    tot_end = 0
    for index,length in rlePairs:
        tot_start = tot_start + index
        tot_end = tot_start + length
        msk[tot_start:tot_end] = MASKCHAR
        tot_start = tot_end
    #msk[2*cols:100*cols] = MASKCHAR # for test display purpose only
    msk = msk.reshape(cols,rows)
    msk = msk.T    
    return msk

def binary_mask_to_rle(binary_mask):
    rle = {'size': list(binary_mask.shape), 'counts': []}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def CropImage(_src, rc):
    x0 = rc[0]
    x1 = rc[0] + rc[2]
    y0 = rc[1]
    y1 = rc[1] + rc[3]
    dst = _src[y0:y1, x0:x1]
    return dst


def crop_to_4v3_or_1v1(_img, _width, _height):
    wd = _width  #int(_width / 32) * 32
    ht = _height #int(_height / 32) * 32    
    ratio = float(_height)/_width
    if ratio > 0.85: # cut to 1:1
        min_dim = min(wd, ht)
        wd = int(min_dim/32)*32
        ht = wd
    else:            # cut to 4:3
        if ratio> 0.75: # cut height
            wd = int(wd/(32*4))*32*4
            ht = int(wd * 3 / 4)
        else : # cut width
            ht = int(ht/(32*3))*32*3
            wd = int(ht * 4 / 3)
    cut_top = int((_height - ht)/2)
    cut_bot = cut_top
    assert(_height == cut_top*2+ ht)
    cut_left = int((_width - wd)/2)
    cut_right = cut_left
    assert(_width == cut_left*2+ wd) 
    rc = (cut_left, cut_top, wd, ht)
    crop_img = CropImage(_img, rc)
    return  crop_img, rc


def transle_polygon(points, rc):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)    
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)== len(ypts))
    left, top, right, bot = rc 
    dxs = xpts - left
    dys = ypts - top
    arr = np.stack((dxs, dys), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return arr, dxs, dys

if __name__ == "__main__":
    #subfolders, files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])
    #for fld in subfolders:
    #    print(fld)

    TOT_IMG_NUM = len(my_imgfiles)
    for i in range(TOT_IMG_NUM):   #for i in range(2):     
        print(i, '/', TOT_IMG_NUM)

        img_filepath = my_imgPaths[i]
        if not os.path.exists(img_filepath):
            continue
        img_file = my_imgfiles[i]

        json_filepath = my_jsonPaths[i]
        if not os.path.exists(json_filepath):
            continue
        tgt_json_filepath = json_filepath.replace(image_root, tgt_root)        

        dataset = json.load(open(json_filepath, 'r'))        
        if not 'imagePath' in dataset:
            continue

        dataset['imagePath'] = img_file

        with open(tgt_json_filepath, "w") as fjs:
            json.dump(dataset, fjs)

        print(json_filepath, '------->', tgt_json_filepath)

    print('Main Done!')

print('All done!')

