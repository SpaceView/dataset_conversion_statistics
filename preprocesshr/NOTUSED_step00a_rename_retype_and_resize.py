"""
功能说明: 获取文件的最后修改时间，并根据这个时间重命名文件，并调整格式为
注意: 
本程序会重命名data_root文件夹下的所有图片文件
本程序不会移动文件的位置(原位重命名)
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

#NOTE： the root dir depends on the dir where PYTHON is executed
os.environ["IMG_ROOT_PATH"] = 'E:/EsightData/metalring/'
os.environ["IMG_TGT_PATH"] = 'E:/EsightData/metalringTgt/'

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

def formatTime(l_time):
    '''格式化时间的函数'''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(l_time))

def nameTime(l_time):
    '''格式化时间的函数'''
    return time.strftime("%Y%m%d%H%M%S",time.localtime(l_time))

def formatByte(number):
    '''格式化文件大小的函数'''
    for(scale, label) in [(1024*1024*1024,"GB"),(1024*1024,"MB"),(1024,"KB")]:
        if number>=scale:
            return  "%.2f %s" %(number*1.0/scale,label)
        elif number ==1:
            return  "1字节"
        else:  #小于1字节
            byte = "%.2f" % (number or 0)
            return (byte[:-3]) if byte.endswith(".00") else byte + "字节"


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
                    print("--------> empty image file (no corresponding annotations): ", img_file)
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
            json_filepath = my_jsonPaths[img_id]

            fileinfo = os.stat(img_filepath)
            """
            print("索引号:",fileinfo.st_ino)
            print("设备名:",fileinfo.st_dev)
            print("文件大小:",formatByte(fileinfo.st_size))
            print("最后一次访问时间:",formatTime(fileinfo.st_atime))
            print("最后一次修改时间:",formatTime(fileinfo.st_mtime))
            print("最后一次状态变化的时间:",fileinfo.st_ctime)
            """
            t_name = '%04d'%(img_name_index)
            t_name = nameTime(fileinfo.st_mtime) + t_name
            t_name = t_name + '.png'
            tgt_imgpath = tgt_root + t_name
            tgt_jsonpath = tgt_imgpath.replace('.png', '.json')

            if not os.path.exists(img_filepath):
                continue

            dataset = json.load(open(json_filepath, 'r'))
            if not 'imagePath' in dataset:
                continue

            img = cv2.imread(img_filepath)
            cv2.imwrite(tgt_imgpath, img)

            #  ---------------------------------------------- 
            # change whatever you need here!
            if(dataset['imagePath'] != t_name):
                #print(img_filepath, '--------------->', t_name)
                dataset['imagePath'] = t_name
            """
            for shape in dataset['shapes']:
                if not (shape['label'] in labels):
                    print(json_filepath, '--------------->', shape['label'])
                if (shape['label'] == 'half'):
                    shape['label'] = 'halfpad'
            """

            with open(tgt_jsonpath, "w") as fjs:
                json.dump(dataset, fjs)
            
            img_name_index = img_name_index + 1            
        
            print('-------------------->{}/{}<--------------------'.format(img_id, TOT_IMG_NUM))

        print('=====================================================')



