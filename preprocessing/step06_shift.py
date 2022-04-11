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
    'E:/EsightData/0221/final/rot',
]

img_root = ''    #os.path.join(data_root, 'images')
ann_root = ''    #os.path.join(data_root, 'coco') 
target_root = 'E:/EsightData/0221/final/transition' #os.path.join(data_root, 'rot')


def RotateImage(_src, angle):
    rad = angle / 180.0 * PI
    sinVal = abs(math.sin(rad))
    cosVal = abs(math.cos(rad))
    width = _src.shape[1]
    height = _src.shape[0]
    target_wd = (int)(width * cosVal + height * sinVal)
    target_ht = (int)(width * sinVal + height * cosVal)
    dx = math.ceil((target_wd - width)/2)
    dx = max(dx, 0) # ensure dx>=0
    dy = math.ceil((target_ht - height)/2)
    dy = max(dy, 0) # ensure dy>=0
    target_wd = dx*2 + width
    target_ht = dy*2 + height
    dx = max(dx,0)
    dy = max(dy,0)
    _dst = cv2.copyMakeBorder(_src, dy, dy, dx, dx, cv2.BORDER_CONSTANT)
    #assert(target_wd == _dst.shape[1])
    #assert(target_ht == _dst.shape[0])
    ptCenter = ( int(target_wd / 2), int(target_ht / 2 ))
    affine_matrix = cv2.getRotationMatrix2D(ptCenter, angle, 1.0)
    _dst = cv2.warpAffine(_dst, affine_matrix, dsize=(target_wd, target_ht), flags = cv2.INTER_NEAREST)
    return _dst, ptCenter


def CropImage(_src, rc):
    x0 = rc[0]
    x1 = rc[0] + rc[2]
    y0 = rc[1]
    y1 = rc[1] + rc[3]
    dst = _src[y0:y1, x0:x1]
    return dst


def shift_image_t(_src, xsht, ysht):
    xabs = abs(xsht)
    yabs = abs(ysht)
    ht = _src.shape[0]
    wd = _src.shape[1]
    dst = copy.deepcopy(_src)
    # first, we copy from _src to dst with x transition
    if xsht < 0:
        dst[0:ht, 0:(wd-xabs)] = _src[0:ht, xabs:wd]
        dst[0:ht, (wd-xabs):wd] = _src[0:ht, 0:xabs]
    else:         
        dst[0:ht, 0:xabs] = _src[0:ht, (wd-xabs):wd]
        dst[0:ht, xabs:wd] = _src[0:ht, 0:(wd-xabs)]
    # then, we copy from dst to _src with y transition
    if ysht < 0:
        _src[0:(ht-yabs), 0:wd] = dst[yabs:ht, 0:wd]
        _src[(ht-yabs):ht, 0:wd] = dst[0:yabs, 0:wd]
    else: 
        _src[0:yabs, 0:wd] = dst[(ht-yabs):ht, 0:wd]
        _src[yabs:ht, 0:wd] = dst[0:(ht-yabs), 0:wd]        
    return _src


def shift_polygon_t(points, xsht, ysht):
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


#NOTE: points must be in format of [[x0, y0], [x1, y1], [x2, y2], ...]
#      ang_rad is the angle in radian
def rotate_polygon_m(points, center, ang_rad):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return DOT(points-center,AR([[COS(ang_rad), SIN(ang_rad)],[-SIN(ang_rad), COS(ang_rad)]]))+center


# equivalent to m, but points can be 
def rotate_polygon_t(points, center, angle):    
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
    XC, YC = center
    dxs = xpts - XC
    dys = ypts - YC
    dxs1 = CV*dxs + SV*dys + XC
    dys1 = -SV*dxs + CV*dys + YC
    arr = np.stack((dxs1, dys1), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return dxs1, dys1, arr


def extract_bbox_t(points):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    bbox = (xpts.min(),ypts.min(),xpts.max(),ypts.max())
    return bbox


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


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def main(): 
    for data_root in data_roots:
        print("=====================> data_root: ", data_root, " <=====================")
        img_root = data_root
        ann_root = data_root
        
        if not osp.exists(img_root):
            continue
        if not osp.exists(ann_root):
            continue
        #target_root = os.path.join(data_root, 'rot')
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
        for idx in range(11854, TOT_IMG_NUM):            
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

            rcbox = [width_org,height_org,0,0]
            for shape in  dataset['shapes']:
                box = extract_bbox_t(shape['points'])
                rcbox[0] = min(box[0], rcbox[0]) #left
                rcbox[1] = min(box[1], rcbox[1]) #top
                rcbox[2] = max(box[2], rcbox[2]) #right
                rcbox[3] = max(box[3], rcbox[3]) #bottom
            rcbox[0] = max(0, rcbox[0])
            rcbox[1] = max(0, rcbox[1])
            rcbox[2] = min(width_org, rcbox[2])
            rcbox[3] = min(height_org, rcbox[3])

            left = int(rcbox[0])
            top = int(rcbox[1])
            xtransition = left + int(width_org-1 - rcbox[2])
            ytransition = top + int(height_org-1 - rcbox[3])

            #===================================================
            # save a rotated json, label(txt), img for each angle
            for shift_id in range(2):
                #NOTE: Don't use shallow copy, e.g. dataset_copy = dataset.copy()
                dataset_copy = copy.deepcopy(dataset)
                img_copy = copy.deepcopy(img_org)

                shapes = dataset_copy['shapes']

                # make new name for the rotated image
                t_sht_img = img_file_name.replace('.png', '-%01d.png'%(shift_id))
                t_sht_json = json_file_name.replace('.json', '-%01d.json'%(shift_id))
                t_sht_txt = json_file_name.replace('.json', '-%01d.txt'%(shift_id))
                path_sht_img = os.path.join(target_root, 'images', t_sht_img)
                path_sht_json = os.path.join(target_root, 'ann', t_sht_json)
                path_sht_label = os.path.join(target_root, 'labels', t_sht_txt)
                flb = open(path_sht_label, "w")
                
                xsht = randint(1, xtransition)
                ysht = randint(1, ytransition)
                xsht = xsht - left
                ysht = ysht - top
                img_sht = shift_image_t(img_copy, xsht, ysht)

                for an in range(len(shapes)):
                    shape = shapes[an]
                    if not ('polygon'==shape['shape_type']):    # -------->label type
                        continue
                    if not 'label' in shape:
                        continue
                    if not 'points' in shape:
                        continue

                    points = shape['points']

                    xpts, ypts, sht_points = shift_polygon_t(points, xsht, ysht)
                    assert(xpts.min()>=0)
                    assert(ypts.min()>=0)
                    assert(xpts.max()<img_sht.shape[1])
                    assert(ypts.max()<img_sht.shape[0])

                    shape['points'] = sht_points.tolist()

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
                retval, encoded_img = cv2.imencode('.png', img_sht)  # Works for '.jpg' as well
                base64_img = base64.b64encode(encoded_img).decode("utf-8")
                dataset_copy['imageData'] = base64_img
                dataset_copy['imageHeight'] = img_sht.shape[0]
                dataset_copy['imageWidth'] = img_sht.shape[1]

                cv2.imwrite(path_sht_img, img_sht)
                with open(path_sht_json, "w") as fjs:
                    json.dump(dataset_copy, fjs)

                
                #print('Done for angle{}'.format(path_sht_img))     
            print('------------------------{}/{}------------------------'.format(idx, TOT_IMG_NUM))

    print('--------> main done <--------')

if __name__ == "__main__":    
    main()


