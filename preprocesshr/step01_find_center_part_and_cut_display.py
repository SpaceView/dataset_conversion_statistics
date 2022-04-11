"""
功能说明: 找到图像的中心点，并对图像进行剪切
注意: 
"""
import numpy as np

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
import sys

import json
import cv2
#import copy
#import math
#from itertools import groupby
#import time
#import shutil
#import random as rng
import base64

#sys.path.insert(1, 'D:/py/dataset_conversion_statistics/')  
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

from img_utils.image_tool import  CropImage, shift_polygon_t

debug_display = False

#NOTE： the root dir depends on the dir where PYTHON is executed
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/metalring/'
os.environ["IMG_TGT_PATH"] = 'E:/EsightData/metalringRC/'

data_root = os.environ['IMG_ROOT_PATH']
tgt_root = os.environ['IMG_TGT_PATH']
if not (os.path.exists(tgt_root)):
    os.makedirs(tgt_root)

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)
    if not (0 == len(subfolders)):
        for dir in list(subfolders):
            sf, f = run_fast_scandir(dir, ext)
            subfolders.extend(sf)
            files.extend(f)
    return subfolders, files


if __name__ == "__main__":
    #NOTE： the root dir depends on the dir where PYTHON is executed
    #       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
    #image_roots = ['E:/EsightData/JX05/02/normal/bot/',
    #    'E:/EsightData/JX05/02/normal/top/',
    #    'E:/EsightData/JX05/03/stoop/top/',
    #    'E:/EsightData/JX05/03/stoop/bot/']

    image_roots, files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])

    if 0==len(image_roots):
        image_roots.append(data_root)

    img_name_index = 0
    for img_root in image_roots:
        ann_root = img_root   # specify image root dir here!!!

        onlyfiles = [f for f in listdir(img_root) if isfile(join(img_root, f)) ]
        my_imgfiles = []
        my_imgPaths = []
        my_jsonfiles = []
        my_jsonPaths = []

        empty_images_count = 0
        for i  in range(len(onlyfiles)):
            sfx = pathlib.Path(onlyfiles[i]).suffix
            if(sfx =='.png') or (sfx =='.jpg') or (sfx =='.jpeg') or (sfx =='.bmp'):
                json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
                #label_file = pathlib.Path(onlyfiles[i]).stem + '.txt'
                json_fp = os.path.join(ann_root, json_file)
                img_fp = os.path.join(img_root, onlyfiles[i])
                if not isfile(json_fp):
                    #os.remove(img_file)
                    print("--------> empty image file (no corresponding annotations): ", img_fp)
                    empty_images_count = empty_images_count + 1
                    continue
                my_imgfiles.append(onlyfiles[i])
                my_imgPaths.append(img_fp)
                my_jsonfiles.append(json_file)
                my_jsonPaths.append(json_fp)

        TOT_IMG_NUM = len(my_imgfiles)
        print('Number of total images: ', TOT_IMG_NUM, ' Number of empty images: ', empty_images_count)        

        for img_id in range(TOT_IMG_NUM):
            #print(img_id, '/', TOT_IMG_NUM)
            img_filepath = my_imgPaths[img_id]
            img_name = my_imgfiles[img_id]
            json_filepath = my_jsonPaths[img_id]
            json_name = my_jsonfiles[img_id]
            #img_dir = os.path.dirname(img_filepath)
            #os.chdir(img_dir)
            
            if not os.path.exists(img_filepath):
                continue
            if not os.path.exists(json_filepath):
                continue

            dataset = json.load(open(json_filepath, 'r'))
            if not 'imagePath' in dataset: 
                continue
            if not 'imageData' in dataset:
                continue
            if not 'imageWidth' in dataset:
                continue
            if not 'imageHeight' in dataset:
                continue
            
            tgt_imgpath = tgt_root + img_name
            fname, fextension = os.path.splitext(tgt_imgpath)
            tgt_imgpath = fname + '.png'
            tgt_jsonpath = tgt_root + json_name

            img = cv2.imread(img_filepath)
            
            img_ht = img.shape[0]
            img_wd = img.shape[1]
            img_half_dim = int(img_ht/2)
            assert(img_ht == 2* img_half_dim)

            # ------------------> find the bounding box to crop <---------------------
            edge = cv2.Canny(img, 80, 255)            
            kernel = np.ones((5, 5), 'uint8')
            edge = cv2.dilate(edge, kernel, iterations=1)

            if debug_display:
                cv2.namedWindow("edge", cv2.WINDOW_NORMAL) #WINDOW_FULLSCREEN) #
                cv2.imshow('edge', edge)
             
            coords = cv2.findNonZero(edge)
            x,y,w,h = cv2.boundingRect(coords)
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            
            box_drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(box_drawing, (x,y), (x+w,y+h), (0,255,0), 2)
            box_drawing = cv2.dilate(box_drawing, kernel, iterations=1)
            
            if debug_display:
                cv2.namedWindow("box_drawing", cv2.WINDOW_NORMAL) #WINDOW_FULLSCREEN) #
                cv2.imshow('box_drawing', box_drawing)
                cv2.waitKey()

            left = center_x - img_half_dim
            right = center_x + img_half_dim
            if(left<0):
                dx = abs(left)
                left = left + dx
                right = right + dx
            elif(right>img_wd):
                dx = right - img_wd
                left = left - dx
                right = right - dx
            assert(left>=0)
            assert(right<=img_wd)
            crop_wd = right - left
            assert(crop_wd == 2*img_half_dim)
            crop_ht = img_ht

            # ------------------> crop image <---------------------
            img_crop = CropImage(img, (left, 0, crop_wd, crop_ht))
            cv2.imwrite(tgt_imgpath, img_crop)

            # ------------------> crop polygons <---------------------
            xsht = -left
            ysht = 0
            
            if(dataset['imagePath'] != img_name):
                dataset['imagePath'] = img_name
            
            for shape in dataset['shapes']:
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
                assert(xpts.max()<img_crop.shape[1])
                assert(ypts.max()<img_crop.shape[0])

                shape['points'] = sht_points.tolist()

            retval, encoded_img = cv2.imencode('.png', img_crop)  # Works for '.jpg' as well
            base64_img = base64.b64encode(encoded_img).decode("utf-8")
            dataset['imageData'] = base64_img
            dataset['imageHeight'] = img_crop.shape[0]
            dataset['imageWidth'] = img_crop.shape[1]

            with open(tgt_jsonpath, "w") as fjs:
                json.dump(dataset, fjs)

            img_name_index = img_name_index + 1
            
            print('-------------------->{}/{}<--------------------'.format(img_id, TOT_IMG_NUM))

        print('=====================================================')

