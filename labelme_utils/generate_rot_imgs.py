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

from img_utils.mask_to_bbox import mask_to_bbox

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
ANG_DIV_NUM = 45      # ---------> for evaluation
ANG_STEP = 360.0/ ANG_DIV_NUM
B_KEEP_WIDTH = True
B_KEEP_HEIGHT = True
REPLACEMENT =  REP_LIST[0]
PI = 3.1415926535897932
MASKCHAR = 255
b_cv2_show = False

from labelme_util_constants import MAX_CAT_ID

#NOTE： the root dir depends on the dir where PYTHON is executed
#os.environ["AREA_IMAGE_PATH"] = 'E:/ESight/GoldLineRn03/'  # ---------> for evaluation
#data_root = os.environ['AREA_IMAGE_PATH']
data_roots = [
    'E:/EsightData/JX05ANN/resize',
    'E:/EsightData/JX05ANN/scale1',
    'E:/EsightData/JX05ANN/scale2',
]

image_root = ''  #os.path.join(data_root, 'images')
ann_root = ''    #os.path.join(data_root, 'coco') 
target_root = '' #os.path.join(data_root, 'rotated')

def RotateImage(_src, angle):
    rad = angle / 180.0 * PI
    sinVal = abs(math.sin(rad))
    cosVal = abs(math.cos(rad))
    width = _src.shape[1]
    height = _src.shape[0]
    target_wd = (int)(width * cosVal + height * sinVal)
    target_ht = (int)(width * sinVal + height * cosVal)
    dx = math.ceil((target_wd - width)/2)
    dy = math.ceil((target_ht - height)/2)
    target_wd = dx*2 + width
    target_ht = dy*2 + height
    dx = max(dx,0)
    dy = max(dy,0)
    _dst = cv2.copyMakeBorder(_src, dy, dy, dx, dx, cv2.BORDER_CONSTANT)
    assert(target_wd == _dst.shape[1])
    assert(target_ht == _dst.shape[0])
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
    xpts = pts[0:len(pts):2]
    ypts = pts[1:len(pts):2]
    assert(len(xpts)==len(ypts))
    XC, YC = center
    dxs = xpts - XC
    dys = ypts - YC
    dxs1 = CV*dxs + SV*dys + XC
    dys1 = -SV*dxs + CV*dys + YC
    arr = np.vstack((dxs1,dys1))
    arrf = arr.flatten('F')
    return dxs1, dys1, arrf

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
        image_root = data_root
        ann_root = data_root        
                
        if not osp.exists(image_root):
            continue
        if not osp.exists(ann_root):
            continue
        target_root = os.path.join(data_root, 'rot')
        if not osp.exists(target_root):
            os.mkdir(target_root)
        tgt_root_img = os.path.join(target_root,'images')
        tgt_root_ann = os.path.join(target_root, 'ann')
        tgt_root_lbl = os.path.join(target_root,'labels')
        if not os.path.exists(tgt_root_img):
            os.makedirs(tgt_root_img)
        if not os.path.exists(tgt_root_ann):
            os.makedirs(tgt_root_ann)
        if not os.path.exists(tgt_root_lbl):
            os.makedirs(tgt_root_lbl)

        empty_images_count = 0

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


        print("empty images without annotation: ", empty_images_count)

        TOT_IMG_NUM = len(my_imgfiles)
        for i in range(TOT_IMG_NUM):   #for i in range(2):     
            #annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
            #if not isfile(annotation_file):
            #    continue
            annotation_file = my_jsonPaths[i]
            json_file = my_jsonfiles[i]
            img_file = my_imgfiles[i]
            dataset = json.load(open(annotation_file, 'r'))
            if not 'annotations' in dataset:
                continue
            if not 'images' in dataset:
                continue

            #annInfo = dataset['annotations']
            imgInfo = dataset['images'][0]

            width_org = imgInfo['width']
            height_org = imgInfo['height']

            print(i, '/', TOT_IMG_NUM)

            path_img_file = my_imgPaths[i]
            if not osp.exists(path_img_file):
                continue

            img_org = cv2.imread(path_img_file)
            assert(img_org.shape[0] == height_org)
            assert(img_org.shape[1] == width_org)

            #===================================================
            # save a rotated json, label(txt), img for each angle
            for ang_id in range(ANG_DIV_NUM):
                #if 0==ang_id:
                #    continue
                
                #NOTE: Don't use shallow copy, e.g. dataset_copy = dataset.copy()
                dataset_copy = copy.deepcopy(dataset)
                img = copy.deepcopy(img_org)

                annInfo_copy = dataset_copy['annotations']

                dAng = ang_id * ANG_STEP
                ptCenter = {0, 0}
                dst, ptCenter = RotateImage(img, dAng)

                img_wd = img.shape[1]
                img_ht = img.shape[0]
                assert(img_wd == width_org)
                assert(img_ht == height_org)
                dst_wd = dst.shape[1]
                dst_ht = dst.shape[0]

                rc_x = 0
                rc_y = 0
                rc_width = dst_wd
                rc_height = dst_ht
                if (B_KEEP_WIDTH):
                    rc_x = (dst_wd - img_wd) / 2
                    rc_width = img_wd                
                if (B_KEEP_HEIGHT):
                    rc_y = (dst_ht - img_ht) / 2
                    rc_height = img_ht
                rc = (int(rc_x), int(rc_y), int(rc_width), int(rc_height))
                img_crop = CropImage(dst, rc)
                
                rc_org = (0, 0, int(rc_width), int(rc_height))

                ptCenter = (ptCenter[0] - int(rc_x), ptCenter[1] - int(rc_y))

                dIntVal = math.floor(dAng)
                dFracVal = (dAng - dIntVal) * 100.0
                t_rot_img = img_file.replace('.png', '-%03d_%02d.png'%(dIntVal, dFracVal))
                t_json = json_file.replace('.json', '-%03d_%02d.json'%(dIntVal, dFracVal))
                t_label = json_file.replace('.json', '-%03d_%02d.txt'%(dIntVal, dFracVal))
                path_rot_img = os.path.join(tgt_root_img, t_rot_img)
                path_json = os.path.join(tgt_root_ann, t_json)
                path_label = os.path.join(tgt_root_lbl, t_label)
                bAreaOutofBoundary = False
                flb = open(path_label, "w")
                for an in range(len(annInfo_copy)):
                    anItem = annInfo_copy[an]
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
                    bbox = []

                    #---------------------------------------------------------
                    #(1) section for json segmentation & bbox file
                    if ('area'==anItem['type']):    # --------> label type                        
                        segment = anItem['segmentation']
                        if not 'counts' in segment:
                            continue

                        rle = segment['counts']
                        mask = rleToMask(rle, img_ht, img_wd)
                        #NOTE: uncomment the below code to see the effect
                        if b_cv2_show:
                            print('cv2 imshow rle')
                            cv2.imshow('rle', mask)
                            cv2.waitKey()
                        
                        #NOTE: ptCenterMsk must be the same as ptCenter
                        mask_rot, ptCenterMsk =  RotateImage(mask, dAng)
                        mask_t = mask_rot.copy()
                        bAreaOutofBoundary = CheckAreaOutofBoundary(mask_t, rc)
                        ptCenterMsk = (ptCenterMsk[0] - int(rc_x), ptCenterMsk[1] - int(rc_y))
                        assert(ptCenter == ptCenterMsk)

                        if (bAreaOutofBoundary) and ('pass' == REPLACEMENT):
                            break  # we don't need this rotated image

                        msk_crop = CropImage(mask_rot, rc)
                        
                        # #NOTE: use the below code
                        # ground_truth_binary_mask = np.array(mask, dtype=np.uint8)
                        # fortran_binary_mask = np.asfortranarray(ground_truth_binary_mask) #can be ommitted
                        # rle = binary_mask_to_rle(fortran_binary_mask)
                        rle = binary_mask_to_rle(msk_crop)

                        #dataset_copy['annotations'][an]['segmentation'] = rle
                        anItem['segmentation'] = rle

                        bx = mask_to_bbox(msk_crop) 

                        # #uncomment the below code to display 
                        # cv2.imshow('msk_crop', msk_crop)
                        # color = (255, 0, 0)
                        # cv2.rectangle(img_crop, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
                        # cv2.imshow('img_crop', img_crop)
                        # cv2.waitKey()
                        anItem['bbox'] = [bx[0], bx[1], bx[2]-bx[0], bx[3]-bx[1]]

                    elif ('polygon'==anItem['type']):
                        points =  anItem['segmentation'][0]
                        center = (int(width_org/2), int(height_org/2))
                        assert(ptCenter == center)
                        xpts, ypts, rot_points = rotate_polygon_t(points, center, dAng)
                        assert(len(xpts)==len(ypts))

                        x1 = xpts.min(); x2 = xpts.max() #x1, x2 = min(xpts), max(xpts)
                        y1 = ypts.min(); y2 = ypts.max()

                        bAreaOutofBoundary = CheckPolygonOutofBoundary(xpts, ypts, rc_org)

                        if (bAreaOutofBoundary) and ('pass' == REPLACEMENT):
                            break  # we don't need this rotated image
                        
                        anItem['segmentation'] = [rot_points.tolist()]
                        anItem['bbox'] = [x1, y1, x2-x1, y2-y1]

                    #---------------------------------------------------------
                    #(2)section for label_txt file
                    bbox = anItem['bbox']
                    bx_left = bbox[0]
                    bx_top = bbox[1]
                    bx_wd = bbox[2]  #NOTE: the mask bbox is (x1, y1, x2, y2)
                    bx_ht = bbox[3]
                    
                    annotation = np.zeros((1, 4)) #annotation = np.zeros((1, 4+2*LMK_NUM))
                    annotation[0, 0] = (bx_left + bx_wd / 2) / width_org  # cx
                    annotation[0, 1] = (bx_top + bx_ht / 2) / height_org  # cy
                    annotation[0, 2] = bx_wd / width_org   # w
                    annotation[0, 3] = bx_ht / height_org  # h

                    #str_label="0 "
                    cat_id = anItem['category_id']-1    #NOTE: here the id starts from 0, thus "-1" is used
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
                if (bAreaOutofBoundary):
                    os.remove(path_label)
                    continue # we don't need this rotated image with the specific rot-angle

                with open(path_json, "w") as fjs:
                    json.dump(dataset_copy, fjs)

                cv2.imwrite(path_rot_img, img_crop)

                print('Done for angle{}'.format(path_rot_img))
            
                print('-----------------------------------------------------------------------')

    print('--------> main done <--------')

if __name__ == "__main__":    
    main()


