
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

#os.environ["LANDMARK_IMAGE_PATH"] = '../Rotated_DONE/'
os.environ["LANDMARK_IMAGE_PATH"] = '../rotated/'
data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, '')        #'db', 'coco', 'images')
ann_root = os.path.join(data_root, 'coco')      # 'db', 'coco', 'instances.json')
yolo_root = os.path.join(data_root, 'yolo')      # 'db', 'coco', 'instances.json')

from os import listdir
from os.path import isfile, join

import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_jsonfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        my_imgfiles.append(onlyfiles[i])
        my_jsonfiles.append(pathlib.Path(onlyfiles[i]).stem + '.json')



for i in range(len(my_imgfiles)):   #for i in range(2): 
    annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
    dataset = json.load(open(annotation_file, 'r'))    
    if not 'annotations' in dataset:
        continue
    if not 'images' in dataset:
        continue

    save_txt_path = os.path.join(yolo_root, my_jsonfiles[i])
    save_txt_path = save_txt_path[:-5] +".txt"

    annInfo = dataset['annotations']
    imgInfo = dataset['images'][0]

    width = imgInfo['width']
    height = imgInfo['height']

    with open(save_txt_path, "w") as f:        
        for an in range(len(annInfo)):
            anItem = annInfo[an]
            if not 'bbox' in anItem:
                continue
            if not 'segmentation' in anItem:
                continue

            bbox = anItem['bbox']
            segment = anItem['segmentation'][0]
            
            if len(bbox)<4:
                continue
            if len(segment)<8:
                continue

            ptx = bbox[0]
            pty = bbox[1]
            wd = bbox[2]
            ht = bbox[3]
            annotation = np.zeros((1, 12))
            annotation[0, 0] = (ptx + wd / 2) / width  # cx
            annotation[0, 1] = (pty + ht / 2) / height  # cy
            annotation[0, 2] = wd / width  # w
            annotation[0, 3] = ht / height  # h

            
            annotation[0, 4] = segment[0] / width  # l0_x
            annotation[0, 5] = segment[1] / height  # l0_y
            annotation[0, 6] = segment[2] / width  # l1_x
            annotation[0, 7] = segment[3]  / height # l1_y
            annotation[0, 8] = segment[4] / width  # l2_x
            annotation[0, 9] = segment[5] / height  # l2_y
            annotation[0, 10] = segment[6] / width  # l3_x
            annotation[0, 11] = segment[7] / height  # l3_y

            str_label="0 "
            for i in range(len(annotation[0])):
                str_label =str_label+" "+str('{:.5f}'.format(annotation[0][i]))
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            print(str_label)
            f.write(str_label)

print('done')

