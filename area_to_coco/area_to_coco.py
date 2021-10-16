"""
@NOTE: Collect all annotations from AutoSeg, 
and combine them into an overall annotation file, in COCO format

@TARGET FORMAT:
{
    "type": "instances",
    "images": [
        {
            "file_name": "0.jpg",
            "height": 600,
            "width": 800,
            "id": 0                    ----> image_id
        }
    ],
    "categories": [
        {
            "supercategory": "none",   ----> supercategory can be anything
            "name": "date",
            "id": 0                    ----> category_id
        },
        {
            "supercategory": "none",
            "name": "hazelnut",
            "id": 2
        },
        {
            "supercategory": "none",
            "name": "fig",
            "id": 1
        }
    ],
    "annotations": [
        {
            "id": 1,                  ----> annotation id (each annotation has a unique id)
            "bbox": [
                100,
                116,
                140,
                170
            ],
            "image_id": 0,
            "segmentation": [],
            "ignore": 0,
            "area": 23800,
            "iscrowd": 0,
            "category_id": 0
        },
        {
            "id": 2,
            "bbox": [
                321,
                320,
                142,
                102
            ],
            "image_id": 0,
            "segmentation": [],
            "ignore": 0,
            "area": 14484,
            "iscrowd": 0,
            "category_id": 0
        }
    ]
}
"""
#
# windows version cocoapi
# https://github.com/philferriere/cocoapi
#
#
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import json
import os
 
from os import listdir
from os.path import isfile, join

import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

import matplotlib as mpl
mpl.use('TkAgg')
import pylab
import matplotlib.rcsetup as rcsetup
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
 

data_type = 'R0805'
os.environ["AREA_IMAGE_PATH"] = 'E:/img/R0805_area/Rotated'
data_root = os.environ['AREA_IMAGE_PATH']
image_root = os.path.join(data_root, '')        # 'db', 'coco', 'images')
ann_root = os.path.join(data_root, 'coco')      # 'db', 'coco', 'instances.json')
annFile='{}/annotations/instances_{}.json'.format(data_root, data_type)

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
my_jsonfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        my_imgfiles.append(onlyfiles[i])
        my_jsonfiles.append(pathlib.Path(onlyfiles[i]).stem + '.json')

"""
"categories": [{
    "supercategory": "none",   ----> supercategory can be anything
    "name": "date",
    "id": 0                    ----> category_id
}]
"""
j_categories = []
cat = {}
cat['supercategory'] = 'none'
cat['name']  = 'R0805P'
cat['id'] = 1  # coco starts with 1 (NOT 0)
j_categories.append(cat)


"""
"images": [
    {
        "file_name": "0.jpg",
        "height": 600,
        "width": 800,
        "id": 0                    ----> image_id
    }
],
"""
j_images = []
for i in range(len(my_imgfiles)):   #for i in range(2): 
    annotation_file = os.path.join(ann_root, my_jsonfiles[i]) 
    dataset = json.load(open(annotation_file, 'r'))
    # collect image file names
    item = {}
    #item['file_name'] =  os.path.join(image_root, my_imgfiles[i])
    item['file_name'] =  my_imgfiles[i]
    item['height'] = dataset['images'][0]['height']
    item['width'] = dataset['images'][0]['width']
    item['id'] = i
    j_images.append(item)
    #print(item)

"""
"annotations": [
    {
        "id": 1,      ----> annotation id (each annotation has a unique id)
        "bbox": [100, 116, 140, 170],
        "image_id": 0,
        "segmentation": [],
        "ignore": 0,
        "area": 23800,
        "iscrowd": 0,
        "category_id": 0
    },
]
"""
unique_ann_id = 0
j_annotations = []
for i in range(len(my_jsonfiles)):   #for i in range(2): 
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
        subitem['id'] = unique_ann_id
        unique_ann_id = unique_ann_id + 1
        j_annotations.append(subitem)

j_result = {}
j_result['images'] = j_images
j_result['annotations'] = j_annotations
j_result['categories']= j_categories

with open(annFile, 'w') as result_file:
     json.dump(j_result, result_file)

print("done!")

