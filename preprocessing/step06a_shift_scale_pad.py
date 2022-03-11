"""
Yolov5 bbox dataformat is: (xc, yc, width, height)
ref. xywh2xyxy or xywhn2xyxy in datasets.py->LoadImagesAndLabels->__getitem__
"""

import json
from cv2 import copyMakeBorder
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
from random import uniform, randint

from img_utils.mask_to_bbox import mask_to_bbox

from scipy import pi as PI
from scipy import array as AR
from scipy import dot as DOT
from scipy import sin as SIN
from scipy import cos as COS
from scipy import rand, arange

from random import randint

#You MUST update these values according to current requirements
from img_utils.labelme_util_constants import MAX_CAT_ID
#TARGET_HEIGHT = 640
#TARGET_WIDTH = 640
from img_utils.labelme_util_constants import TRAIN_IMG_WDITH as TARGET_WIDTH
from img_utils.labelme_util_constants import TRAIN_IMG_HEIGHT as TARGET_HEIGHT
from img_utils.labelme_util_constants import TRAIN_SCALE_LO as SCALE_LO
from img_utils.labelme_util_constants import TRAIN_SCALE_HI as SCALE_HI


PI = 3.1415926535897932
MASKCHAR = 255
EDGE_BUFFER = 10
REQUIRE_SCALING = True

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
    #'E:/EsightData/0221/final/imgResize/rot'
    'E:/EsightData/0221/final/rot',    
    #'E:/EsightData/0221/final/test',
]

img_root = ''    #os.path.join(data_root, 'images')
ann_root = ''    #os.path.join(data_root, 'coco') 
target_root = 'E:/EsightData/0221/final/rotrans' #os.path.join(data_root, 'rot')

def shift_image_t(_src, xsht, ysht):
    xabs = abs(xsht)
    yabs = abs(ysht)
    ht = _src.shape[0]
    wd = _src.shape[1]
    dst = np.zeros(_src.shape, np.uint8)
    # first, we copy from _src to dst with x transition
    if xsht < 0:
        dst[0:ht, 0:(wd-xabs)] = _src[0:ht, xabs:wd]
        dst[0:ht, (wd-xabs):wd] = _src[0:ht, 0:xabs]
    elif xsht>0:         
        dst[0:ht, 0:xabs] = _src[0:ht, (wd-xabs):wd]
        dst[0:ht, xabs:wd] = _src[0:ht, 0:(wd-xabs)]
    # then, we copy from dst to _src with y transition
    if ysht < 0:
        _src[0:(ht-yabs), 0:wd] = dst[yabs:ht, 0:wd]
        _src[(ht-yabs):ht, 0:wd] = dst[0:yabs, 0:wd]
    elif ysht>0: 
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


def pad_image_t(_src, _width, _height):
    ht = _src.shape[0]
    wd = _src.shape[1]
    assert(_height >= ht)
    assert(_width >= wd)
    t_top = int((_height - ht)/2)
    t_bot = _height - ht - t_top
    t_left = int((_width - wd)/2)
    t_right = _width - wd - t_left
    t_img = cv2.copyMakeBorder(src=_src, top=t_top, bottom=t_bot, left=t_left, right=t_right,
                                borderType=cv2.BORDER_CONSTANT,value=[114, 114, 114])
    return t_img, (t_left, t_top, t_right, t_bot)


def extract_bbox_t(points):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    bbox = (xpts.min(),ypts.min(),xpts.max(),ypts.max())
    return bbox


def extract_ann_bbox_t(shapes, _width, _height):
    rcbb = [_width, _height,0,0]
    for shape in  shapes:
        box = extract_bbox_t(shape['points'])
        rcbb[0] = min(box[0], rcbb[0]) #left
        rcbb[1] = min(box[1], rcbb[1]) #top
        rcbb[2] = max(box[2], rcbb[2]) #right
        rcbb[3] = max(box[3], rcbb[3]) #bottom
    rcbb[0] = max(0, rcbb[0])
    rcbb[1] = max(0, rcbb[1])
    rcbb[2] = min(_width, rcbb[2])
    rcbb[3] = min(_height, rcbb[3])
    return rcbb

def scale_image_at_random_scale_t(_src, _annbbox, _tgt_wd, _tgt_ht):
    #(1) resize image at some random scale
    wd = _src.shape[1]
    ht = _src.shape[0]
    wdr = _annbbox[2] - _annbbox[0] + 2*EDGE_BUFFER  # we need some edge for buffering
    htr = _annbbox[3] - _annbbox[1] + 2*EDGE_BUFFER
    xrup = min(float(wd)/wdr, SCALE_HI)
    yrup = min(float(ht)/htr, SCALE_HI)
    #xdir = randint(0,1)
    #ydir = randint(0,1)
    #if(xdir):  # 1 --> enlarge
    #    xscale = uniform(1.0, xrup)
    #else:      # 0 --> reduce
    #    xscale = uniform(0.75, 1.0)
    #if(ydir):
    #    yscale = uniform(1.0, yrup)
    #else:
    #    yscale = uniform(0.75, 1.0)        
    rup = min(xrup, yrup)
    dir = randint(0,1)
    if(dir):
        scale = uniform(1.0, rup)
    else:
        scale = uniform(SCALE_LO, 1.0)
    xscale = scale
    yscale = scale            
    dest_ht = int(ht * yscale)
    dest_wd = int(wd * xscale)
    dim = (dest_wd, dest_ht)
    img_rsz = cv2.resize(_src, dim, interpolation=cv2.INTER_AREA)
    anbox_rsz = [_annbbox[0]*xscale, _annbbox[1]*yscale, _annbbox[2]*xscale, _annbbox[3]*yscale]
    # (2) pad the image ensure (width > target_width) and (height > target_height)
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0
    if(dest_ht < _tgt_ht):
        pad_top = int((_tgt_ht - dest_ht)/2)
        pad_bot = _tgt_ht - dest_ht - pad_top
    if(dest_wd < _tgt_wd):
        pad_left = int((_tgt_wd - dest_wd)/2)
        pad_right = _tgt_wd - dest_wd - pad_left
    t_img = cv2.copyMakeBorder(src=img_rsz, top=pad_top, bottom=pad_bot, left=pad_left, right=pad_right,
                                borderType=cv2.BORDER_CONSTANT,value=[114, 114, 114])
    # (3) Crop the annotation area to target dimension
    xcenter = int((anbox_rsz[0]+anbox_rsz[2])/2)
    ycenter = int((anbox_rsz[1]+anbox_rsz[3])/2)
    crop_x0 = min(int(xcenter - _tgt_wd/2), int(t_img.shape[1] - _tgt_wd))
    crop_y0 = min(int(ycenter - _tgt_ht/2), int(t_img.shape[0] - _tgt_ht))
    crop_x0 = max(0, crop_x0)
    crop_y0 = max(0, crop_y0)
    crop_x1 = crop_x0 + _tgt_wd
    crop_y1 = crop_y0 + _tgt_ht
    img_crop = t_img[crop_y0:crop_y1, crop_x0:crop_x1]    
    scale_pad_cut_info = ([xscale, yscale], [pad_left, pad_top], [crop_x0, crop_y0])
    if(img_crop.shape[0]!=640):
        print('error ht')
    if(img_crop.shape[1]!=640):
        print('error wd')
    return img_crop, scale_pad_cut_info

def scale_polygon_t(points, scale_info):
    scl = scale_info[0]  #[xscale, yscale]
    pad = scale_info[1]  #[t_left, t_top] for pad
    cut = scale_info[2]  #[x0, y0] for cut
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)==len(ypts))
    dxs = xpts*scl[0] + pad[0] - cut[0]
    dys = ypts*scl[1] + pad[1] - cut[1]
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
        #target_root = os.path.join(data_root, 'rot')
        target_root_img = os.path.join(target_root, 'images')
        target_root_ann = os.path.join(target_root, 'ann')
        if not osp.exists(target_root):
            os.mkdir(target_root)
        if not osp.exists(target_root_img):
            os.mkdir(target_root_img)
        if not osp.exists(target_root_ann):
            os.mkdir(target_root_ann)
        
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
        for idx in range(0, TOT_IMG_NUM):            
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

            #===================================================
            # save a rotated json, label(txt), img for each angle
            for shift_id in range(2):
                #NOTE: Don't use shallow copy, e.g. dataset_copy = dataset.copy()
                dataset_copy = copy.deepcopy(dataset)
                img_copy = copy.deepcopy(img_org)
                shapes = dataset_copy['shapes']

                # (1) first, we pad all images (from 640x480) to 640x640)
                # --(1a) scale + pad + cut for 1 image
                if (1 == shift_id) and (REQUIRE_SCALING):  
                    rcbb = extract_ann_bbox_t(shapes, width_org, height_org)
                    img_pad, scale_info = scale_image_at_random_scale_t(img_copy, rcbb, TARGET_WIDTH, TARGET_HEIGHT)
                    for shape in shapes:
                        if not 'points' in shape:
                            continue
                        xpts, ypts, arr = scale_polygon_t(shape['points'], scale_info)
                        shape['points'] = arr.tolist()
                # --(1b) pad the rest images
                elif (height_org != TARGET_HEIGHT) or (width_org != TARGET_WIDTH):
                    img_pad, rc_pad = pad_image_t(img_copy, TARGET_WIDTH, TARGET_HEIGHT)
                    for shape in  shapes:
                        xpts, ypts, arr = shift_polygon_t(shape['points'], rc_pad[0], rc_pad[1])
                        shape['points'] = arr.tolist()
                else:
                    img_pad = img_copy

                # (2) find the bounding box (rcbb) of all annotations
                annbbox = extract_ann_bbox_t(shapes, TARGET_WIDTH, TARGET_HEIGHT)
                
                bb_left = max(0, int(annbbox[0] - 5))       # keep 5 pixels around the annotation
                bb_top = max(0, int(annbbox[1] - 5))
                bb_right = max(0, int(width_org-1 -5 - annbbox[2]))
                bb_bot = max(0, int(height_org-1 - 5- annbbox[3]))
                
                xtransition = bb_left + bb_right
                ytransition = bb_top + bb_bot

                # make new name for the shifted image
                t_sht_img = img_file_name.replace('.png', '-%01d.png'%(shift_id))
                t_sht_json = json_file_name.replace('.json', '-%01d.json'%(shift_id))
                #t_sht_txt = json_file_name.replace('.json', '-%01d.txt'%(shift_id))
                path_sht_img = os.path.join(target_root_img, t_sht_img)
                path_sht_json = os.path.join(target_root_ann, t_sht_json)
                #path_sht_label = os.path.join(target_root, 'labels', t_sht_txt)
                #flb = open(path_sht_label, "w")
                
                xsht = randint(0, xtransition)
                ysht = randint(0, ytransition)
                xsht = xsht - bb_left
                ysht = ysht - bb_top
                img_sht = shift_image_t(img_pad, xsht, ysht)

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

                    """
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
                    """

                #flb.close()
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


