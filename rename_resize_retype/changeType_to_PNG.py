
import json
import numpy as np
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from collections import defaultdict
import sys

import os
from os import listdir
from os.path import isfile, join
import pathlib

import cv2

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["LANDMARK_IMAGE_PATH"] = '../../img/'         # ---------> for training
#os.environ["LANDMARK_IMAGE_PATH"] = '../../R0805P_eval/'  # ---------> for evaluation

data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, 'QrCode/photos/PadCropTo640x480')
target_root = os.path.join(data_root, 'QrCode/photos/PadCropTo640x480png')

if not os.path.exists(target_root):
    os.makedirs(target_root)

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_tgtfiles = []
for i  in range(len(onlyfiles)):
    impath = pathlib.Path(onlyfiles[i])
    if (impath.suffix=='.png') or (impath.suffix=='.jpg') or (impath.suffix=='.jpeg'):          
        my_imgfiles.append(os.path.join(image_root, onlyfiles[i]))
        my_tgtfiles.append(os.path.join(target_root, impath.stem + '.png'))

TOT_IMG_NUM = len(my_imgfiles)
for i in range(TOT_IMG_NUM):
    img_path = my_imgfiles[i]
    img = cv2.imread(img_path)    

    cv2.imwrite(my_tgtfiles[i], img)

    print(i, '/', TOT_IMG_NUM)

   
print('done')

