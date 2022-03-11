"""
Yolov5 bbox dataformat is: (xc, yc, width, height)
ref. xywh2xyxy or xywhn2xyxy in datasets.py->LoadImagesAndLabels->__getitem__
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
import os.path as osp
from os import listdir
from os.path import isfile, join
import pathlib  # usage: print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'
import sys

#sys.path.insert(1, 'D:/py/img_python/')
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

import cv2
import math
from itertools import groupby

import base64

from img_utils.mask_to_bbox import mask_to_bbox

from scipy import pi as PI
from scipy import array as AR
from scipy import dot as DOT
from scipy import sin as SIN
from scipy import cos as COS
from scipy import rand, arange


PI = 3.1415926535897932
MASKCHAR = 255
b_cv2_show = False

from random import randint

#You MUST update these values according to current requirements
from img_utils.labelme_util_constants import MAX_CAT_ID

labels = {
    'ok': 0,
    'noline':1,
    'multi':2,
    'dirt':3,
    'halfpad':4,
    'disorder':5,
    'twist':6,
    'offpad':7,
    'lugpad':8,
    'tailpad':9,
    'rotpad':10,
}

#NOTE： the root dir depends on the dir where PYTHON is executed
#os.environ["AREA_IMAGE_PATH"] = 'E:/ESight/GoldLineRn03/'  # ---------> for evaluation
#data_root = os.environ['AREA_IMAGE_PATH']
data_roots = [
    'E:/EsightData/0221/final/imgResizePad',
    #'E:/EsightData/0221/final/transition'
]

img_root = ''    #os.path.join(data_root, 'images')
ann_root = ''    #os.path.join(data_root, 'coco') 
target_root = '' #os.path.join(data_root, 'rot')


def extract_bbox_t(points):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    bbox = (xpts.min(),ypts.min(),xpts.max(),ypts.max())
    return bbox


def main(): 
    for data_root in data_roots:
        print("=====================> data_root: ", data_root, " <=====================")
        img_root = os.path.join(data_root, 'images')
        ann_root = os.path.join(data_root, 'ann')
        
        if not osp.exists(img_root):
            continue
        if not osp.exists(ann_root):
            continue
        target_root = os.path.join(data_root, 'labels')
        if not osp.exists(target_root):
            os.mkdir(target_root)

        empty_images_count = 0

        onlyfiles = [f for f in listdir(img_root) if isfile(join(img_root, f)) ]
        my_imgfiles = []
        my_imgPaths = []
        my_jsonfiles = []
        my_jsonPaths = []

        for i  in range(len(onlyfiles)):
            if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
                json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
                label_file = pathlib.Path(onlyfiles[i]).stem + '.txt'
                annotation_file = os.path.join(ann_root, json_file)
                img_file = os.path.join(img_root, onlyfiles[i])
                if not isfile(annotation_file):              
                    #os.remove(img_file)
                    print("--------> empty image file (no corresponding annotations): ", img_file)
                    empty_images_count = empty_images_count + 1
                    continue
                my_imgfiles.append(onlyfiles[i])
                my_imgPaths.append(img_file)
                my_jsonfiles.append(json_file)        
                my_jsonPaths.append(annotation_file)

        print("empty images without annotation: ", empty_images_count)

        TOT_IMG_NUM = len(my_imgfiles)
        for idx in range(TOT_IMG_NUM):            
            #annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
            #if not isfile(annotation_file):
            #    continue
            path_img_file = my_imgPaths[idx]
            img_file_name = my_imgfiles[idx]
            path_json_file = my_jsonPaths[idx]
            json_file_name = my_jsonfiles[idx]

            if not osp.exists(path_img_file):
                continue
            if not osp.exists(path_json_file):
                continue

            dataset = json.load(open(path_json_file, 'r'))
            if not 'shapes' in dataset:
                continue
            if 0==len(dataset['shapes']):
                continue
            if not 'imageHeight' in dataset:
                continue
            if not 'imageWidth' in dataset:
                continue
            if not 'imageData' in dataset:
                continue
           
            height_org = dataset['imageHeight']
            width_org = dataset['imageWidth']
            
            #===================================================
            t_txt = json_file_name.replace('.json', '.txt')
            path_sht_label = os.path.join(target_root, t_txt)
            flb = open(path_sht_label, "w")
            
            shapes = dataset['shapes']            

            for an in range(len(shapes)):
                shape = shapes[an]
                if not ('polygon'==shape['shape_type']):    # -------->label type
                    continue
                if not 'label' in shape:
                    continue
                if not 'points' in shape:
                    continue

                pts = shape['points']
                if not isinstance(pts, np.ndarray):        
                    pts = np.array(pts)                    
                l_pts = len(pts)
                xpts = pts[0:l_pts, 0]
                ypts = pts[0:l_pts, 1]                

                # -------- label file --------
                x1 = xpts.min(); x2 = xpts.max() #x1, x2 = min(xpts), max(xpts)
                y1 = ypts.min(); y2 = ypts.max()                    
                bx_left = x1  #bbox = [x1, y1, x2-x1, y2-y1]                    
                bx_top = y1
                bx_wd = x2-x1
                bx_ht = y2-y1
                
                annotation = np.zeros((1, 4)) #annotation = np.zeros((1, 4+2*LMK_NUM))
                annotation[0, 0] = (bx_left + bx_wd / 2) / width_org  # cx
                annotation[0, 1] = (bx_top + bx_ht / 2) / height_org  # cy
                annotation[0, 2] = bx_wd / width_org   # w
                annotation[0, 3] = bx_ht / height_org  # h

                #str_label="0 "
                cat_id = labels[shape['label']]    #NOTE: here the id starts from 0, thus "-1" is used
                if cat_id<0 or cat_id>=MAX_CAT_ID:
                    print("ERROR: the category id is out of range")

                str_label='{} '.format(cat_id)
                
                for i in range(len(annotation[0])):
                    str_label =str_label+" "+str('{:.5f}'.format(annotation[0][i]))
                str_label = str_label.replace('[', '').replace(']', '')
                str_label = str_label.replace(',', '') + '\n'
                #print(str_label)
                flb.write(str_label)

            flb.close()
                
                #print('Done for angle{}'.format(path_sht_img))     
            print('------------------------{}/{}------------------------'.format(idx, TOT_IMG_NUM))

    print('--------> main done <--------')

if __name__ == "__main__":    
    main()


