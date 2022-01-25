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

CROP_WIDTH = 4000
CROP_HEIGHT = 3000
TARGET_WIDTH = 640
TARGET_HEIGHT = 480  #600*600/800
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)
PI = 3.1415926535897932
MASKCHAR = 255
crop_req = False
resize_req = True

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = 'E:/ESight/GoldLineRn02/'
os.environ["TARGET_PATH"] = 'E:/ESight/GoldLineRn03/'

data_root = os.environ['IMG_ROOT_PATH']
image_root = data_root
ann_root = os.path.join(data_root, 'coco')

target_root = os.environ['TARGET_PATH']
target_ann_root = os.path.join(target_root, 'coco')

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

if not os.path.exists(target_root):
    os.makedirs(target_root)

if not os.path.exists(target_ann_root):
    os.makedirs(target_ann_root)

#===========================================================================

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

def resize_image(img_in, interpolation):
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
    img_resized = cv2.resize(img_in, dim, interpolation=interpolation)
    return img_resized


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

def CropImage(_src, rc):
    x0 = rc[0]
    x1 = rc[0] + rc[2]
    y0 = rc[1]
    y1 = rc[1] + rc[3]
    dst = _src[y0:y1, x0:x1]
    return dst

def binary_mask_to_rle(binary_mask):
    rle = {'size': list(binary_mask.shape), 'counts': []}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


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
        tgt_filepath = img_filepath.replace(image_root, target_root)

        json_filepath = my_jsonPaths[i]
        if not os.path.exists(json_filepath):
            continue
        tgt_jsonpath = json_filepath.replace(image_root, target_root)

        dataset = json.load(open(json_filepath, 'r'))        
        if not 'annotations' in dataset:
            continue
        if not 'images' in dataset:
            continue
        
        annInfo = dataset['annotations']
        imgInfo = dataset['images'][0]
        img_width = imgInfo['width']
        img_height = imgInfo['height']

        #==================================================================
        #(SECTION II) image resize
        img = cv2.imread(img_filepath)
        if(crop_req):
            img = img[0:CROP_HEIGHT, 0:CROP_WIDTH, :]
        img = resize_image(img, cv2.INTER_AREA)
        cv2.imwrite(tgt_filepath, img)

        #==================================================================
        #(SECTION I) annotation resize
        for an in range(len(annInfo)):
            anItem = annInfo[an]
            if not 'bbox' in anItem:
                continue
            if not 'segmentation' in anItem:
                continue
            if not 'type' in anItem:
                continue            
            if not ('area'==anItem['type']):    # -------->label type
                continue
            
            if not 'category_id' in anItem:
                continue

            bbox_org = anItem['bbox']
            if len(bbox_org)<4:
                continue

            if not 'counts' in anItem['segmentation']:
                continue

            rle_org = anItem['segmentation']['counts']
            mask = rleToMask(rle_org, img_height, img_width)
            #NOTE: uncomment the below code to see the effect
            #cv2.imshow('rlemask', mask)
            #cv2.waitKey()

            if(crop_req):
                rc = (0,0, TARGET_WIDTH, TARGET_HEIGHT)
                mask = CropImage(mask, rc)

            mask = resize_image(mask, cv2.INTER_NEAREST)
            # #NOTE: use the below code
            # ground_truth_binary_mask = np.array(mask, dtype=np.uint8)
            # fortran_binary_mask = np.asfortranarray(ground_truth_binary_mask) #can be ommitted
            # rle = binary_mask_to_rle(fortran_binary_mask)
            rle = binary_mask_to_rle(mask)
            anItem['segmentation'] = rle
            
            #---------------------------------------------------------
            #(2)section for label_txt file
            bbox = mask_to_bbox(mask)
            bbox[2] = bbox[2] - bbox[0] # wd = x1 - x0
            bbox[3] = bbox[3] - bbox[1] # ht = y1 - y0
            anItem['bbox'] = bbox
            """
            # #uncomment the below code to display 
            dmsk = copy.deepcopy(mask)
            imsk = copy.deepcopy(img)
            redImg = np.zeros(imsk.shape, imsk.dtype)
            redImg[:,:] = (0, 0, 255)
            redMask = cv2.bitwise_and(redImg, redImg, mask=dmsk)
            cv2.addWeighted(redMask, 1, img, 1, 0, imsk)
            cv2.imshow('image masked', imsk)
            cv2.waitKey()
            #"""

        with open(tgt_jsonpath, "w") as fjs:
            json.dump(dataset, fjs)

        print('--------------------------------------')

    print('Main Done!')

print('All done!')

