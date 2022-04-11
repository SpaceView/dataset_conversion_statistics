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
from os import listdir
from os.path import isfile, join
import pathlib

import cv2
import base64

import math
from itertools import groupby

import sys
#sys.path.insert(1, 'D:/py/dataset_conversion_statistics/')  
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

from img_utils.mask_to_bbox import mask_to_bbox

#CROP_WIDTH = 4000
#CROP_HEIGHT = 3000
#TARGET_WIDTH = 600
#TARGET_HEIGHT = 450  #600*600/800
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)
PI = 3.1415926535897932
MASKCHAR = 255
#crop_req = False
#resize_req = True


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
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/0221/final/imgResize'

data_root = os.environ['IMG_ROOT_PATH']
image_root = data_root
ann_root = data_root            #os.path.join(data_root, 'coco')

tgt_root = os.path.join(data_root, 'checked')
if not (os.path.exists(tgt_root)):
        os.makedirs(tgt_root)

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

#===========================================================================

if __name__ == "__main__":
    TOT_IMG_NUM = len(my_imgfiles)
    for i in range(TOT_IMG_NUM):   #for i in range(2):     
        print(i, '/', TOT_IMG_NUM)

        img_filepath = my_imgPaths[i]
        if not os.path.exists(img_filepath):
            continue
        img_file = my_imgfiles[i]

        json_filepath = my_jsonPaths[i]
        if not os.path.exists(json_filepath):
            continue
        tgt_json_filepath = json_filepath.replace(image_root, tgt_root)        

        dataset = json.load(open(json_filepath, 'r'))        
        if not 'imagePath' in dataset:
            continue
        
        # change whatever you need
        if(dataset['imagePath'] != img_file):
            print(img_filepath, '--------------->', img_file)
            dataset['imagePath'] = img_file
        for shape in dataset['shapes']:
            if not (shape['label'] in labels):
                print(json_filepath, '--------------->', shape['label'])
            if (shape['label'] == 'half'):
                shape['label'] = 'halfpad'

        with open(tgt_json_filepath, "w") as fjs:
            json.dump(dataset, fjs)

        #print(json_filepath, '------->', tgt_json_filepath)

    print('Main Done!')

print('All done!')

