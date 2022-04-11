"""
功能说明: 显示最外缘的框
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
#sys.path.insert(1, 'D:/py/dataset_conversion_statistics/')  
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

import json
import copy
import cv2
import math
from itertools import groupby
import time
import shutil
import random as rng


#NOTE： the root dir depends on the dir where PYTHON is executed
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/metalring/'

data_root = os.environ['IMG_ROOT_PATH']

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
            img_file = my_imgfiles[img_id]
            json_file = my_jsonfiles[img_id]
            img_dir = os.path.dirname(img_filepath)
            os.chdir(img_dir)
            
            if not os.path.exists(img_file):
                continue

            img = cv2.imread(img_filepath)
            
            ht = img.shape[0]
            wd = img.shape[1]        #img_rsz = cv2.resize
            edge = cv2.Canny(img, 80, 255)
            
            kernel = np.ones((15, 15), 'uint8')
            edge = cv2.dilate(edge, kernel, iterations=1)
            cv2.namedWindow("edge", cv2.WINDOW_NORMAL) #WINDOW_FULLSCREEN) #
            cv2.imshow('edge', edge)
             
            #cv2.waitKey()

            #edge = cv2.dilate(src=edge, kernel=cv2.Mat(), anchor=cv2.Point(-1,-1), iterations=1)
            contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            minRect = [None]*len(contours)
            minEllipse = [None]*len(contours)
            for i, c in enumerate(contours):
                minRect[i] = cv2.minAreaRect(c)
                if c.shape[0] > 5:
                    minEllipse[i] = cv2.fitEllipse(c)

            drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
            for i, c in enumerate(contours):
                #color = (255, 255, 255) #
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                # contour
                cv2.drawContours(drawing, contours, i, color, 3)
                # ellipse
                if c.shape[0] > 5:
                    cv2.ellipse(drawing, minEllipse[i], color, 3)
                # rotated rectangle
                box = cv2.boxPoints(minRect[i])
                box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.drawContours(drawing, [box], 0, color, 3)

            #drawing = cv2.dilate(drawing, kernel, iterations=1)
            
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL) #WINDOW_FULLSCREEN) #
            cv2.imshow('Contours', drawing)
             
            cv2.waitKey()

            

            print('-------------------->{}/{}<--------------------'.format(img_id, TOT_IMG_NUM))

        print('=====================================================')