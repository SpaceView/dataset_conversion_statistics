
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


"""
dbface format structure
# xxxx.jpg
bbx bby bbw bbh lmx1 lmy1 _v1 lmx2 lmy2 _v2 lmx3 lmy3 _v3 lmx4 lmy4 _v4 lmx5 lmy5 _v5
bbx bby bbw bbh lmx1 lmy1 _v1 lmx2 lmy2 _v2 lmx3 lmy3 _v3 lmx4 lmy4 _v4 lmx5 lmy5 _v5
# yyyy.jpg
bbx bby bbw bbh lmx1 lmy1 _v1 lmx2 lmy2 _v2 lmx3 lmy3 _v3 lmx4 lmy4 _v4 lmx5 lmy5 _v5
bbx bby bbw bbh lmx1 lmy1 _v1 lmx2 lmy2 _v2 lmx3 lmy3 _v3 lmx4 lmy4 _v4 lmx5 lmy5 _v5
bbx bby bbw bbh lmx1 lmy1 _v1 lmx2 lmy2 _v2 lmx3 lmy3 _v3 lmx4 lmy4 _v4 lmx5 lmy5 _v5

generally values are given this way (DBFACE doesn't use this value)
_v1, _v2, _v3, _v4 = 1: ok, 0: invisible, -1: invalid
"""

is_train_data = 0

#os.environ["LANDMARK_IMAGE_PATH"] = '../Rotated_DONE/'
if is_train_data:
    os.environ["LANDMARK_IMAGE_PATH"] = '../rotated/'   #for dbface training
else:
    os.environ["LANDMARK_IMAGE_PATH"] = '../test/'      #for dbface test
data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, '')        #'db', 'coco', 'images')
ann_root = os.path.join(data_root, 'coco')      # 'db', 'coco', 'instances.json')
yolo_root = os.path.join(data_root, 'dbface')      # 'db', 'coco', 'instances.json')

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


save_label_txt_path = ''
if is_train_data:
    save_label_txt_path = os.path.join(image_root, 'label/train_label.txt')
else:
    save_label_txt_path = os.path.join(image_root, 'label/test_label.txt')    
    
f_label_txt = open(save_label_txt_path, 'w')
if None == f_label_txt:
    print("label text open failed")

for i in range(len(my_imgfiles)):   #for i in range(2): 
    annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
    dataset = json.load(open(annotation_file, 'r'))    
    if not 'annotations' in dataset:
        continue
    if not 'images' in dataset:
        continue

    f_label_txt.write('# ')
    f_label_txt.write(my_imgfiles[i])
    f_label_txt.write('\n')

    save_txt_path = os.path.join(yolo_root, my_jsonfiles[i])
    save_txt_path = save_txt_path[:-5] +".txt"

    annInfo = dataset['annotations']
    imgInfo = dataset['images'][0]

    width = imgInfo['width']
    height = imgInfo['height']
    
    if( 0 == i%10):
        print(i, ' / ' , len(my_imgfiles))

    with open(save_txt_path, "w") as f:        
        for an in range(len(annInfo)):
            anItem = annInfo[an]
            if not 'bbox' in anItem:
                continue
            if not 'segmentation' in anItem:
                continue

            bbox = anItem['bbox']
            segment = anItem['segmentation'][0]
            
            val_step = 2
            annotation = np.zeros((1, 12))
            if len(bbox)<4:
                continue
            if 12 == len(segment):
                val_step = 3
                annotation = np.zeros((1, 16))
            elif 8 == len(segment):
                val_step = 2
            else:
                continue

            ptx = bbox[0]
            pty = bbox[1]
            wd = bbox[2]
            ht = bbox[3]
           
            #annotation[0, 0] = (ptx + wd / 2) / width  # cx
            #annotation[0, 1] = (pty + ht / 2) / height  # cy
            annotation[0, 0] = ptx / width  # x
            annotation[0, 1] = pty / height # y
            annotation[0, 2] = wd / width   # w
            annotation[0, 3] = ht / height  # h
            if(3 == val_step):
                val_id = 0
                annotation[0, 4] = segment[val_id] / width  # l0_x
                annotation[0, 5] = segment[val_id+1] / height  # l0_y
                annotation[0, 6] = segment[val_id+2]
                val_id += val_step
                annotation[0, 7] = segment[val_id] / width  # l1_x
                annotation[0, 8] = segment[val_id+1]  / height # l1_y
                annotation[0, 9] = segment[val_id+2]
                val_id += val_step
                annotation[0, 10] = segment[val_id] / width  # l2_x
                annotation[0, 11] = segment[val_id+1] / height  # l2_y
                annotation[0, 12] = segment[val_id+2]
                val_id += val_step
                annotation[0, 13] = segment[val_id] / width  # l3_x
                annotation[0, 14] = segment[val_id+1] / height  # l3_y
                annotation[0, 15] = segment[val_id+2]
            else: #2== val_step                
                val_id = 0
                annotation[0, 4] = segment[val_id] / width  # l0_x
                annotation[0, 5] = segment[val_id+1] / height  # l0_y
                val_id += val_step
                annotation[0, 6] = segment[val_id] / width  # l1_x
                annotation[0, 7] = segment[val_id+1]  / height # l1_y
                val_id += val_step
                annotation[0, 8] = segment[val_id] / width  # l2_x
                annotation[0, 9] = segment[val_id+1] / height  # l2_y
                val_id += val_step
                annotation[0, 10] = segment[val_id] / width  # l3_x
                annotation[0, 11] = segment[val_id+1] / height  # l3_y
                
            #str_label="0 "
            str_label = str( '{:.5f}'.format(annotation[0][0]) )
            for i in range(1, len(annotation[0])):
                str_label =str_label+" "+str('{:.5f}'.format(annotation[0][i]))
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            #print(str_label)
            f.write(str_label)  
            f_label_txt.write(str_label)
        f.close()          

f_label_txt.close()

print('done')

