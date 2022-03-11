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
from img_utils.labelme_util_constants import TRAIN_IMG_HEIGHT as TARGET_HEIGHT
from img_utils.labelme_util_constants import TRAIN_IMG_WDITH  as TARGET_WIDTH

from scipy import pi as PI
from scipy import array as AR
from scipy import dot as DOT
from scipy import sin as SIN
from scipy import cos as COS
from scipy import rand, arange


#replacement setting:
#  0: pass (no replace, we don't use this rotated image if any area is out of the boundary)
#  1: polygon with landmarks (for future use only)
#  2: annotation bbox (for future use only)
#  3: scaled template based on landmarks (for future use only)
REP_LIST = ['pass', 'polygon', 'bbox', 'temp']
#change the below 
ANG_DIV_NUM = 8      # ---------> for evaluation
ANG_START = 25
ANG_STEP = 360 / ANG_DIV_NUM
B_KEEP_WIDTH = True
B_KEEP_HEIGHT = True
REPLACEMENT =  REP_LIST[0]
PI = 3.1415926535897932
MASKCHAR = 255
b_cv2_show = False

from random import randint

#NOTE： the root dir depends on the dir where PYTHON is executed
#os.environ["AREA_IMAGE_PATH"] = 'E:/ESight/GoldLineRn03/'  # ---------> for evaluation
#data_root = os.environ['AREA_IMAGE_PATH']
data_roots = [
    'E:/EsightData/0221/final/imgResize',
]

img_root = ''   #os.path.join(data_root, 'images')
ann_root = ''   #os.path.join(data_root, 'coco') 
target_root ='' #os.path.join(data_root, 'rot')

def pad_image_t(_src, _tgt_wd, _tgt_ht):
    width = _src.shape[1]
    height = _src.shape[0]
    target_wd = _tgt_wd
    target_ht = _tgt_ht
    left = math.floor((target_wd - width)/2)
    assert(left>=0) #dx = max(dx, 0) # ensure dx>=0
    top = math.floor((target_ht - height)/2)
    assert(top>=0) #dy = max(dy, 0) # ensure dy>=0
    right = target_wd - width - left
    bot = target_ht - height - top
    dst_img = cv2.copyMakeBorder(_src, top, bot, left, right, cv2.BORDER_CONSTANT)
    assert(target_wd == dst_img.shape[1])
    assert(target_ht == dst_img.shape[0])
    pad = (left, top)
    return dst_img, pad


def pad_polygon_t(points, pad):
    xsht, ysht = pad
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)==len(ypts))
    dxs = xpts + xsht
    dys = ypts + ysht
    arr = np.stack((dxs, dys), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return dxs, dys, arr

def main(): 
    for data_root in data_roots:
        print("=====================> data_root: ", data_root, " <=====================")
        img_root = data_root
        ann_root = data_root

        if not osp.exists(img_root):
            continue
        if not osp.exists(ann_root):
            continue
        target_root = os.path.join(data_root, 'pad')
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
        for i in range(TOT_IMG_NUM): 
            #annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
            #if not isfile(annotation_file):
            #    continue
            path_img_file = my_imgPaths[i]
            img_file_name = my_imgfiles[i]
            path_json_file = my_jsonPaths[i]
            json_file_name = my_jsonfiles[i]

            if not osp.exists(path_img_file):
                continue
            if not osp.exists(path_json_file):
                continue

            img_org = cv2.imread(path_img_file)

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

            assert(img_org.shape[0] == height_org)
            assert(img_org.shape[1] == width_org)

            #===================================================
            # save a rotated json, label(txt), img for each angle

            dataset_copy = dataset  #copy.deepcopy(dataset)
            img_copy = img_org      #copy.deepcopy(img_org)
            assert(img_copy.shape[1] == width_org)
            assert(img_copy.shape[0] == height_org)
            shapes = dataset_copy['shapes']
            
            dst_img, dst_pad = pad_image_t(img_copy, TARGET_WIDTH, TARGET_HEIGHT)

            # make new name for the rotated image                
            t_pad_img = img_file_name.replace('.png', '-pad.png')
            t_pad_json = json_file_name.replace('.json', '-pad.json')
            path_pad_img = os.path.join(target_root, t_pad_img)
            path_pad_json = os.path.join(target_root, t_pad_json)

            for an in range(len(shapes)):
                shape = shapes[an]
                if not ('polygon'==shape['shape_type']):    # -------->label type
                    continue
                if not 'label' in shape:
                    continue
                if not 'points' in shape:
                    continue

                points = shape['points']

                # 说明：旋转时和前面的图片旋转保持一致
                xpts, ypts, pad_points = pad_polygon_t(points, dst_pad)
                assert(len(xpts)==len(ypts))                
                
                shape['points'] = pad_points.tolist()
            
            retval, encoded_img = cv2.imencode('.png', dst_img)  # Works for '.jpg' as well
            base64_img = base64.b64encode(encoded_img).decode("utf-8")
            dataset_copy['imageData'] = base64_img
            dataset_copy['imageHeight'] = dst_img.shape[0]
            dataset_copy['imageWidth'] = dst_img.shape[1]
            dataset_copy['imagePath'] = t_pad_img

            cv2.imwrite(path_pad_img, dst_img)
            with open(path_pad_json, "w") as fjs:
                json.dump(dataset_copy, fjs)
            
            #print(path_pad_img)
            #print(path_pad_json)                
            print('------------------------{}/{}------------------------'.format(i, TOT_IMG_NUM))
            #print(i, '/', TOT_IMG_NUM)                       

    print('--------> main done <--------')

if __name__ == "__main__":    
    main()


