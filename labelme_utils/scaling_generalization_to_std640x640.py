"""
Scaling images from 640x640 to 
(1) 600x600 ---> padding to 640x480
(2) 600x640 
(3) 640x600
(4) 540x540 ---> padding to 640x480   
(5) 540x640
(6) 640x540

Scaling images from 640x480 to 
(1) 600x450 ---> padding to 640x480
(2) 600x480 
(3) 640x450
(4) 540x400 ---> padding to 640x480   
(5) 540x480
(6) 640x450
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

WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)
PI = 3.1415926535897932
MASKCHAR = 255

# You can use 640x640 or 640x480
TARGET_WIDTH = 640
TARGET_HEIGHT = 640  
#resize_req = True

# resized+padding ---> save to target_path_1
WD_1 = 600
HT_1 = 600

# resized+padding ---> save to target_path_2
WD_2 = 540
HT_2 = 540

RSZ_NUM = 3  # 3 image for [WD_1, HT_1] & [WD_2, HT_2]
TGT_DIM = [[[HT_1, WD_1], [HT_1, TARGET_WIDTH], [TARGET_HEIGHT, WD_1]], 
           [[HT_2, WD_2], [HT_2, TARGET_WIDTH], [TARGET_HEIGHT, WD_2]]]

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/JX05ANN/resize'
os.environ["TARGET_PATH_1"] = 'E:/EsightData/JX05ANN/scale1'
os.environ["TARGET_PATH_2"] = 'E:/EsightData/JX05ANN/scale2'

data_root = os.environ['IMG_ROOT_PATH']
image_root = data_root 
ann_root = data_root   #os.path.join(data_root, 'coco')

target_roots = [os.environ['TARGET_PATH_1'], os.environ['TARGET_PATH_2']]
target_ann_roots = []  #os.path.join(target_roots[0], 'coco'), os.path.join(target_roots[1], 'coco')
for rt in target_roots:
    anrt = rt #os.path.join(rt, 'coco')
    if not os.path.exists(rt):
        os.makedirs(rt)
    if not os.path.exists(anrt):
        os.makedirs(anrt)
    target_ann_roots.append(anrt)

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

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files

def resize_image_to_targetsize(img_in, interpolation):
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

def resize_polygon(dim, target_dim, points):
    wd, ht = dim
    target_wd, target_ht = target_dim
    ry = target_ht / ht
    rx = target_wd / wd
    pts = np.array(points)
    xpts = pts[0:len(pts):2]
    ypts = pts[1:len(pts):2]
    assert(len(xpts)==len(ypts))
    d_ypts = ypts * ry
    d_xpts = xpts * rx
    arr = np.vstack((d_xpts,d_ypts))
    arrf = arr.flatten('F')
    return d_xpts, d_ypts, arrf


def pad_polygon(dim, paddings, points):
    left, top, right, bottom = paddings
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    xpts = pts[0:len(pts):2]
    ypts = pts[1:len(pts):2]
    assert(len(xpts)==len(ypts))
    d_ypts = ypts + top
    d_xpts = xpts + left
    arr = np.vstack((d_xpts,d_ypts))
    arrf = arr.flatten('F')
    return d_xpts, d_ypts, arrf


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
    TOT_ROOT_NUM = len(target_roots)
    for rt_id in range(TOT_ROOT_NUM):
        tgt_root = target_roots[rt_id]

        for img_id in range(TOT_IMG_NUM):
            print(img_id, '/', TOT_IMG_NUM)

            img_filepath = my_imgPaths[img_id]
            if not os.path.exists(img_filepath):
                continue
            tgt_fpath = img_filepath.replace(image_root, tgt_root)

            json_filepath = my_jsonPaths[img_id]
            if not os.path.exists(json_filepath):
                continue
            tgt_jpath = json_filepath.replace(image_root, tgt_root)

            dataset_org = json.load(open(json_filepath, 'r'))
            if not 'annotations' in dataset_org:
                continue
            if not 'images' in dataset_org:
                continue

            img_org = cv2.imread(img_filepath)
            
            for t_id in range(RSZ_NUM):
                dataset = copy.deepcopy(dataset_org)

                t_height = TGT_DIM[rt_id][t_id][0]
                t_width = TGT_DIM[rt_id][t_id][1]
                t_dim = (t_width, t_height)
                t_padx = int((TARGET_WIDTH - t_width)/2)
                assert (2*t_padx + t_width == TARGET_WIDTH)
                t_pady = int((TARGET_HEIGHT - t_height)/2)
                assert (2*t_pady + t_height == TARGET_HEIGHT)
                
                t_ex = '_{}x{}'.format(t_width, t_height) 
                t_suffix = pathlib.Path(tgt_fpath).suffix
                t_name = tgt_fpath.replace(t_suffix, t_ex+t_suffix)
                j_name = tgt_jpath.replace('.json', t_ex+'.json') 
                
                #==================================================================
                #(SECTION II) image resize
                img_rsz = cv2.resize(src=img_org, dsize=t_dim, interpolation=cv2.INTER_AREA)
                img_cpy = cv2.copyMakeBorder(src=img_rsz, top=t_pady, bottom=t_pady, left=t_padx, right=t_padx, 
                                       borderType=cv2.BORDER_CONSTANT,value=[114, 114, 114])
                cv2.imwrite(t_name, img_cpy)

                #==================================================================
                #(SECTION I) annotation resize
                annInfo = dataset['annotations']
                imgInfo = dataset['images'][0]
                img_width_org = imgInfo['width']
                img_height_org = imgInfo['height']
                for an in range(len(annInfo)):
                    anItem = annInfo[an]
                    if not 'bbox' in anItem:
                        continue
                    if not 'segmentation' in anItem:
                        continue
                    if not 'type' in anItem:
                        continue
                    
                    if not 'category_id' in anItem:
                        continue

                    bbox_org = anItem['bbox']
                    if len(bbox_org)<4:
                        continue

                    #if not 'counts' in anItem['segmentation']:
                    #    continue
                    if ('area'==anItem['type']):    # -------->label type
                        
                        rle_org = anItem['segmentation']['counts']
                        mask = rleToMask(rle_org, img_height_org, img_width_org)
                        """
                        # NOTE: uncomment the below code to see the effect
                        cv2.imshow('rlemask', mask)
                        cv2.waitKey()
                        #"""
                        
                        mask_rsz = cv2.resize(src=mask, dsize=t_dim, interpolation=cv2.INTER_NEAREST)
                        mask_cpy = cv2.copyMakeBorder(src=mask_rsz, top=t_pady, bottom=t_pady, left=t_padx, right=t_padx, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

                        # #NOTE: use the below code
                        # ground_truth_binary_mask = np.array(mask, dtype=np.uint8)
                        # fortran_binary_mask = np.asfortranarray(ground_truth_binary_mask) #can be ommitted
                        # rle = binary_mask_to_rle(fortran_binary_mask)                    
                        rle = binary_mask_to_rle(mask_cpy)
                        anItem['segmentation'] = rle
                        
                        #---------------------------------------------------------
                        #(2)section for label_txt file
                        bbox = mask_to_bbox(mask_cpy)
                        bbox[2] = bbox[2] - bbox[0] # wd = x1 - x0
                        bbox[3] = bbox[3] - bbox[1] # ht = y1 - y0
                        anItem['bbox'] = bbox
                        """
                        # #uncomment the below code to display 
                        dmsk = copy.deepcopy(mask_cpy)
                        imsk = copy.deepcopy(img_cpy)
                        redImg = np.zeros(imsk.shape, imsk.dtype)
                        redImg[:,:] = (0, 0, 255)
                        redMask = cv2.bitwise_and(redImg, redImg, mask=dmsk)
                        cv2.addWeighted(redMask, 1, imsk, 1, 0, imsk)
                        cv2.imshow('image masked', imsk)
                        cv2.waitKey()
                        #"""
                    elif ('polygon'==anItem['type']):
                        points =  anItem['segmentation'][0]
                        xpts, ypts, rsz_points = resize_polygon((img_width_org, img_height_org), t_dim, points)
                        assert(len(xpts)==len(ypts))

                        pxpts, pypts, pad_points  = pad_polygon(t_dim, (t_padx, t_pady, t_padx, t_pady), rsz_points)

                        #rsz_points = [v for v in zip(xpts,ypts)]                
                        x1 = pxpts.min(); x2 = pxpts.max() #x1, x2 = min(xpts), max(xpts)
                        y1 = pypts.min(); y2 = pypts.max()

                        anItem['segmentation'] = [pad_points.tolist()]
                        anItem['bbox'] = [x1, y1, x2-x1, y2-y1]

                imgInfo['width'] = TARGET_WIDTH
                imgInfo['height'] = TARGET_HEIGHT
                with open(j_name, "w") as fjs:
                    json.dump(dataset, fjs)

            print('-------------------->{}<--------------------'.format(img_id))

        print('==============================>TARGET ROOT SEPERATOR<==============================')

    print('Main Done!')

print('All done!')

