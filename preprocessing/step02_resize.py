from asyncio.windows_events import NULL
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
import base64

TARGET_HEIGHT = 640
TARGET_WIDTH = 640
WHITE_PAD = (255,255,255)
BLACK_PAD = (0,0,0)

img_org_root = 'E:/EsightData/0218test/img/'
img_tgt_root = 'E:/EsightData/0218test/img_resize'


#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
os.environ["IMG_ROOT_PATH"] = img_org_root
os.environ["TARGET_PATH"] = img_tgt_root

data_root = os.environ['IMG_ROOT_PATH']
target_root = os.environ['TARGET_PATH']

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


"""
#NOTE: if you don't need to list folders recursively, use the below code
onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
image_root = os.path.join(data_root, 'subsoure')
target_root = os.path.join(data_root, 'subtarget')
"""

if not os.path.exists(target_root):
    os.makedirs(target_root)


def resize_image(_img):
    ht, wd = _img.shape[0:2]
    ratio = float(ht)/wd
    if (not (ht == wd)) and (not (0.75 == ratio)):
        return None
    dest_ht = TARGET_HEIGHT #640
    dest_wd = TARGET_WIDTH  #640
    if not (ht == wd): # wd:ht == 4:3
        dest_ht = int(TARGET_HEIGHT * 3 / 4)
    dim = (dest_wd, dest_ht)
    img_resized = cv2.resize(_img, dim, interpolation=cv2.INTER_AREA)
    return img_resized

def resize_to_4v3_or_1v1(_img):
    ht, wd = _img.shape[0:2]
    ratio = float(ht)/wd
    wd = TARGET_WIDTH  #640
    if ratio > 0.85: # --> to 1:1        
        ht = TARGET_HEIGHT #640        
    else:            # --> to 4:3
        ht = int(TARGET_HEIGHT * 3 / 4)
    dim = (wd, ht)
    img_resized = cv2.resize(_img, dim, interpolation=cv2.INTER_AREA)
    return img_resized


def resize_polygon(points,  ht0, wd0, ht1, wd1):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)    
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)== len(ypts))
    rx = float(wd1/wd0)
    ry = float(ht1/ht0) 
    dxs = xpts * rx
    dys = ypts * ry
    arr = np.stack((dxs, dys), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return arr, dxs, dys

if __name__ == "__main__":
    subfolders, img_files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])
    json_subfolders, json_files = run_fast_scandir(data_root, [".json"])

    # make sure that each image file has a corresponding json file
    if not (len(img_files) == len(json_files)):
        for fid in range(len(img_files)):
            img_name = os.path.splitext(os.path.basename(img_files[fid]))[0]
            json_name = os.path.splitext(os.path.basename(json_files[fid]))[0]
            if not (img_name == json_name):                
                print(os.path.basename(img_files[fid]), ' doesnot match corresponding json file ', json_files[fid])
                break
        print('ERROR: number of image files must match nubmer of json files')
        exit()

    if not (subfolders==json_subfolders):
        print('ERROR: subfolders doesnot match')
        exit()

    for img_root_fld in subfolders:
        print(img_root_fld)
        
        onlyfiles = [f for f in listdir(img_root_fld) if isfile(join(img_root_fld, f)) ]
        my_imgfiles = []
        my_imgPaths = []
        my_jsonfiles = []
        my_jsonPaths = []

        for i  in range(len(onlyfiles)):
            if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
                json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
                annotation_file = os.path.join(img_root_fld, json_file)
                img_file = os.path.join(img_root_fld, onlyfiles[i])
                if not isfile(annotation_file):
                    #os.remove(img_file)
                    print("--------> empty image file (no corresponding annotations): ", img_file)
                    empty_images_count = empty_images_count + 1
                    continue
                my_imgfiles.append(onlyfiles[i])
                my_imgPaths.append(img_file)
                my_jsonfiles.append(json_file)        
                my_jsonPaths.append(annotation_file)
                
        TOT_IMG_NUM = len(my_imgfiles)
        for i in range(TOT_IMG_NUM):            
            img_filepath = my_imgPaths[i]
            if not os.path.exists(img_filepath):
                continue
            tgt_imgpath = img_filepath.replace(img_root_fld, img_tgt_root)

            json_filepath = my_jsonPaths[i]
            if not os.path.exists(json_filepath):
                continue
            tgt_jsonpath = json_filepath.replace(img_root_fld, img_tgt_root)
            
            tgt_imgpath = tgt_imgpath.rsplit( ".", 1 )[0] + '.png' # conver to png if NOT png file            
            #dir = os.path.dirname(fpng)
            #if not (os.path.exists(dir)):
            #    os.makedirs(dir)

            img = cv2.imread(img_filepath)
            if img is None:
                continue
            
            dataset = json.load(open(json_filepath, 'r'))        
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
            
            shapes = dataset['shapes']
            img_height = dataset['imageHeight']
            img_width = dataset['imageWidth']

            assert(img_height == img.shape[0])
            assert(img_width == img.shape[1])
            
            img_rsz = resize_image(img)
            if img_rsz is None:
                img_rsz = resize_to_4v3_or_1v1(img)
            if img_rsz is None:
                continue

            for an in range(len(shapes)):
                shape = shapes[an]
                if not ('polygon'==shape['shape_type']):    # -------->label type
                    continue
                if not 'label' in shape:
                    continue
                if not 'points' in shape:
                    continue

                points = shape['points']
                arrpts, xpts, ypts = resize_polygon(points, img_height, img_width, img_rsz.shape[0], img_rsz.shape[1])
                shape['points'] = arrpts.tolist()

            retval, encoded_img = cv2.imencode('.png', img_rsz)  # Works for '.jpg' as well
            base64_img = base64.b64encode(encoded_img).decode("utf-8")
            dataset['imageData'] = base64_img
            #dataset['shapes'] = shapes
            dataset['imageHeight'] = img_rsz.shape[0]
            dataset['imageWidth'] = img_rsz.shape[1]

            cv2.imwrite(tgt_imgpath, img_rsz)
            with open(tgt_jsonpath, "w") as fjs:
                json.dump(dataset, fjs)

            print(tgt_imgpath)
            print(tgt_jsonpath)
            print('--------------------------------------')
        
    print('Main Done!')

print('All done!')


