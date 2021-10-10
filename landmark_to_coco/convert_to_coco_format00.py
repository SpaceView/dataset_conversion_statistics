
import json
#import time
import numpy as np
import copy
#import itertools
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
 
import os
from collections import defaultdict
import sys

os.environ["LANDMARK_IMAGE_PATH"] = '/home/mc/devAI/R0805/'
data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, 'imgorg')        #'db', 'coco', 'images')
ann_root = os.path.join(data_root, 'coco')                 # 'db', 'coco', 'instances.json')

from os import listdir
from os.path import isfile, join

import pathlib
print(pathlib.Path('yourPath.example').suffix) # '.example'

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_jsonfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        my_imgfiles.append(onlyfiles[i])
        my_jsonfiles.append(pathlib.Path(onlyfiles[i]).stem + '.json')

#
#"categories": [{"name": "background", "id": 0}, 
#                               {"name": "R0805P", "id": 1}]}
#
j_categories = []
cat = {}
cat['name']  = 'background'
cat['id'] = 0
j_categories.append(cat)
cat = {}
cat['name']  = 'R0805P'
cat['id'] = 1
j_categories.append(cat)


#"images": 
#	[
#		{"file_name": "./image/path/name.jpg", "height": 683, "width": 1024, "id": 0},
#		...
#		{"file_name": "./image/path/name.jpg", "height": 683, "width": 1024, "id": 0},		
#	], 
j_images = []
for i in range(2): #(len(my_imgfiles)):
    annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
    dataset = json.load(open(annotation_file, 'r'))
    # collect image file names
    item = {}
    item['file_name'] =  os.path.join(image_root, my_imgfiles[i])
    item['height'] = dataset['images'][0]['height']
    item['width'] = dataset['images'][0]['width']
    item['id'] = i
    j_images.append(item)
    print(item)

#"annotations": 
#  [
#       {"segmentation": [[421.1, 133.2, 445.5, 125.5, 432.6, 145.4, 432.3, 159.9, 450.6, 153.4]], "area": 4356.76, "iscrowd": 0, "image_id": 0, "bbox": [411.5, 100.1, 59.6, 73.1], "category_id": 1, "id": 0, "ignore": 0}, 
#  ], 
ann_id = 0
j_annotations = []
for i in range(2): #(len(my_jsonfiles)):
    annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
    dataset = json.load(open(annotation_file, 'r'))
    anns = dataset['annotations']
    for  k in range(len(anns)):
        subitem = {}
        subitem['segmentation'] = anns[k]['segmentation']
        subitem['area'] = anns[k]['bbox'][2] * anns[k]['bbox'][3]
        subitem['iscrowd'] = 0
        subitem['image_id'] = i
        subitem['bbox'] = anns[k]['bbox']
        subitem['category_id'] = 1
        subitem['id'] = ann_id
        ann_id = ann_id + 1
        j_annotations.append(subitem)

j_result = {}
j_result['images'] = j_images
j_result['annotations'] = j_annotations
j_result['categories']= j_categories

file_path = 'combination_R0805P.json'
with open(file_path, 'w') as result_file:
     json.dump(j_result, result_file)

print(1)

