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

img_root = ''    #os.path.join(data_root, 'images')
ann_root = ''    #os.path.join(data_root, 'coco') 
target_root = '' #os.path.join(data_root, 'rot')


def rotate_image_t(_src, _angle, _tgt_wd, _tgt_ht):
    rad = _angle / 180.0 * PI
    sinVal = abs(math.sin(rad))
    cosVal = abs(math.cos(rad))
    width = _src.shape[1]
    height = _src.shape[0]
    target_wd = (int)(width * cosVal + height * sinVal)
    target_wd = max(target_wd, _tgt_wd)
    target_ht = (int)(width * sinVal + height * cosVal)
    target_ht = max(target_ht, _tgt_ht)    
    dx = math.ceil((target_wd - width)/2)
    assert(dx>=0) #dx = max(dx, 0) # ensure dx>=0
    dy = math.ceil((target_ht - height)/2)
    assert(dy>=0) #dy = max(dy, 0) # ensure dy>=0
    target_wd = dx*2 + width
    target_ht = dy*2 + height
    dst_img = cv2.copyMakeBorder(_src, dy, dy, dx, dx, cv2.BORDER_CONSTANT)
    #assert(target_wd == _dst.shape[1])
    #assert(target_ht == _dst.shape[0])
    ptCenter = ( int(target_wd / 2), int(target_ht / 2 ))
    pad = (dx, dy)
    affine_matrix = cv2.getRotationMatrix2D(ptCenter, _angle, 1.0)
    dst_img = cv2.warpAffine(dst_img, affine_matrix, dsize=(target_wd, target_ht), flags = cv2.INTER_NEAREST)
    return dst_img, ptCenter, pad


def crop_image_t(_src, rc):
    x0 = rc[0]
    x1 = rc[0] + rc[2]
    y0 = rc[1]
    y1 = rc[1] + rc[3]
    dst = _src[y0:y1, x0:x1]
    return dst


#NOTE: points must be in format of [[x0, y0], [x1, y1], [x2, y2], ...]
#      ang_rad is the angle in radian
def rotate_polygon_m(points, center, ang_rad):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return DOT(points-center,AR([[COS(ang_rad), SIN(ang_rad)],[-SIN(ang_rad), COS(ang_rad)]]))+center


# equivalent to m, but points can be 
# 说明：旋转时一定要以图片的正中心为中心点旋转, 和前面的图片旋转保持一致
def rotate_shift_polygon_t(points, pad, center, angle, crp):    
    rad = angle / 180.0 * PI
    SV = math.sin(rad)
    CV = math.cos(rad)
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)==len(ypts))
    xpts = xpts + pad[0]
    ypts = ypts + pad[1]
    XC, YC = center
    dxs = xpts - XC
    dys = ypts - YC
    dxs1 = CV*dxs + SV*dys + XC
    dys1 = -SV*dxs + CV*dys + YC
    dxs1 = dxs1 - crp[0]
    dys1 = dys1 - crp[1]
    arr = np.stack((dxs1, dys1), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return dxs1, dys1, arr

def CheckAreaOutofBoundary(msk, rc):
    #msk_wd = msk.shape[1]
    #msk_ht = msk.shape[0]
    rc_x = rc[0]
    rc_y = rc[1]
    rc_wd = rc[2]
    rc_ht = rc[3]

    y_low = int(rc_y)
    y_up =  int(rc_y + rc_ht)
    x_low =  int(rc_x)
    x_up =  int(rc_x + rc_wd)

    msk[y_low:y_up, x_low:x_up] = 0
    msk_bool = (msk == MASKCHAR)
    count = np.sum(msk_bool)
    if(count):
        return True  #NOTE: we found masked pixels out of the rect (rc) bounday
    return False


def CheckPolygonOutofBoundary(xpts, ypts, rc):
    #msk_wd = msk.shape[1]
    #msk_ht = msk.shape[0]
    rc_x = rc[0]
    rc_y = rc[1]
    rc_wd = rc[2]
    rc_ht = rc[3]

    y_low = int(rc_y)
    y_up =  int(rc_y + rc_ht)
    x_low =  int(rc_x)
    x_up =  int(rc_x + rc_wd)

    ybool = (ypts<y_low) | (ypts>=y_up)
    xbool = (xpts<x_low) | (xpts>=x_up)
    count = np.sum(ybool) + np.sum(xbool)
    if(count):
        return True  #NOTE: we found masked pixels out of the rect (rc) bounday
    return False

def main(): 
    for data_root in data_roots:
        print("=====================> data_root: ", data_root, " <=====================")
        img_root = data_root
        ann_root = data_root
        
        if not osp.exists(img_root):
            continue
        if not osp.exists(ann_root):
            continue
        target_root = os.path.join(data_root, 'rot')
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
            ANG_START = randint(1, ANG_STEP)
            for ang_id in range(ANG_DIV_NUM):
                #if 0==ang_id:
                #    continue
                #NOTE: Don't use shallow copy, e.g. dataset_copy = dataset.copy()
                dataset_copy = copy.deepcopy(dataset)
                img_copy = copy.deepcopy(img_org)
                assert(img_copy.shape[1] == width_org)
                assert(img_copy.shape[0] == height_org)
                
                shapes = dataset_copy['shapes']

                dAng = ANG_START + ang_id * ANG_STEP # double value Angle
                #dAng = 80.0 for test of 640x480 rotated (width will be narrowed down)
                dst_img, dst_center, dst_pad = rotate_image_t(img_copy, dAng, TARGET_WIDTH, TARGET_HEIGHT)

                dst_wd = dst_img.shape[1]
                dst_ht = dst_img.shape[0]

                crp_x = max(0, (dst_wd - TARGET_WIDTH) / 2)
                crp_y = max(0, (dst_ht - TARGET_HEIGHT) / 2)
                rc = (int(crp_x), int(crp_y), TARGET_WIDTH, TARGET_HEIGHT)
                img_crop = crop_image_t(dst_img, rc)
                
                rc_target = (0, 0, TARGET_WIDTH, TARGET_HEIGHT)

                # make new name for the rotated image                
                dIntVal = math.floor(dAng)
                t_rot_img = img_file_name.replace('.png', '-%03d.png'%(dIntVal))
                t_rot_json = json_file_name.replace('.json', '-%03d.json'%(dIntVal))
                path_rot_img = os.path.join(target_root, t_rot_img)
                path_rot_json = os.path.join(target_root, t_rot_json)

                bAreaOutofBoundary = False
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
                    xpts, ypts, rot_points = rotate_shift_polygon_t(points, dst_pad, dst_center, dAng, (crp_x, crp_y))
                    assert(len(xpts)==len(ypts))
                    
                    bAreaOutofBoundary = CheckPolygonOutofBoundary(xpts, ypts, rc_target)

                    if (bAreaOutofBoundary) and ('pass' == REPLACEMENT):
                        break  # we don't need this rotated image                   

                    shape['points'] = rot_points.tolist()
                
                if (bAreaOutofBoundary):
                    continue # we don't need this rotated image with the specific rot-angle

                retval, encoded_img = cv2.imencode('.png', img_crop)  # Works for '.jpg' as well
                base64_img = base64.b64encode(encoded_img).decode("utf-8")
                dataset_copy['imageData'] = base64_img
                dataset_copy['imageHeight'] = img_crop.shape[0]
                dataset_copy['imageWidth'] = img_crop.shape[1]

                cv2.imwrite(path_rot_img, img_crop)
                with open(path_rot_json, "w") as fjs:
                    json.dump(dataset_copy, fjs)
                
                #print(path_rot_img)
                #print(path_rot_json)                
                print('Done for angle{}'.format(path_rot_img))     
            print('------------------------{}/{}------------------------'.format(i, TOT_IMG_NUM))
            #print(i, '/', TOT_IMG_NUM)                       

    print('--------> main done <--------')

if __name__ == "__main__":    
    main()


