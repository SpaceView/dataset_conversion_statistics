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

import os
from collections import defaultdict
import sys

from labelme_util_constants import MAX_CAT_ID

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
#os.environ["POLYAREA_IMAGE_PATH"] = 'E:/ESight/GoldLineRename/rotated/'  # ---------> for training
os.environ["POLYAREA_IMAGE_PATH"] = 'E:/EsightData/JX05ANN/generated/'  # ---------> for evaluation

data_root = os.environ['POLYAREA_IMAGE_PATH']

image_root = os.path.join(data_root, 'PNGImages')        #'db', 'coco', 'images'
ann_root = os.path.join(data_root, 'ann') 
#label_root = os.path.join(data_root, 'labels')
label_root = os.path.join(data_root, 'labels')

from os import listdir
from os.path import isfile, join

import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

empty_images_count = 0

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_imgPaths = []
my_jsonfiles = []
my_jsonPaths = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
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
    dataset = json.load(open(annotation_file, 'r'))    
    if not 'annotations' in dataset:
        continue
    if (not 'images' in dataset):
        continue
    
    save_txt_path = os.path.join(label_root, my_jsonfiles[i])
    save_txt_path = save_txt_path[:-5] +".txt"

    annInfo = dataset['annotations']
    imgInfo = dataset['images'][0]

    width = imgInfo['width']
    height = imgInfo['height']

    print(i, '/', TOT_IMG_NUM)

    with open(save_txt_path, "w") as f:        
        for an in range(len(annInfo)):
            anItem = annInfo[an]
            if not 'bbox' in anItem:
                continue
            if not 'segmentation' in anItem:
                continue
            if not 'type' in anItem:
                continue            
            if not ('polygon'==anItem['type']):    # -------->label type
                continue
            
            if not 'category_id' in anItem:
                continue

            bbox = anItem['bbox']
            #segment = anItem['landmarks']  #segment = anItem['segmentation'][0]
            
            if len(bbox)<4:
                continue
            #if len(segment)<(2*LMK_NUM):
            #    continue

            # convert AutoSeg BBOX [left, top, width, height] to YOLO format
            bx_left = bbox[0]
            bx_top = bbox[1]
            bx_wd = bbox[2]
            bx_ht = bbox[3]
            annotation = np.zeros((1, 4)) #annotation = np.zeros((1, 4+2*LMK_NUM))
            annotation[0, 0] = (bx_left + bx_wd / 2) / width  # cx
            annotation[0, 1] = (bx_top + bx_ht / 2) / height  # cy
            annotation[0, 2] = bx_wd / width  # w
            annotation[0, 3] = bx_ht / height  # h

            # # record landmark positions [x, y, val, x, y, val, x, y, val]
            # for lmkid in range(LMK_NUM):
            #     annotation[0, 4+2*lmkid] = segment[3*lmkid] / width     # landmark_x
            #     annotation[0, 5+2*lmkid] = segment[3*lmkid+1] / height  # landmark_x

            #str_label="0 "
            cat_id = anItem['category_id']-1
            if cat_id<0 or cat_id>=MAX_CAT_ID:
                print("ERROR: the category id is out of range")

            str_label='{} '.format(cat_id)
            
            for i in range(len(annotation[0])):
                str_label =str_label+" "+str('{:.5f}'.format(annotation[0][i]))
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            #print(str_label)
            f.write(str_label)

print('done')

