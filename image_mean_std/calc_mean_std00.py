

"""
# test read flag from opencv
flag = [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE] 
img_name = os.path.join(image_root, my_imgfiles[0])    
for f in flag:
    img = cv2.imread(img_name, f)
    print(type(img))
    print(img.shape)
    #print(img)
    cv2.imshow('image window',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

"""
# test square 
x = torch.randint(10,[3,2,3])
y=x**2
x
tensor([[[8, 9, 4],
         [3, 4, 6]],

        [[3, 4, 5],
         [3, 5, 0]],

        [[6, 0, 9],
         [4, 3, 4]]])
y
tensor([[[64, 81, 16],
         [ 9, 16, 36]],

        [[ 9, 16, 25],
         [ 9, 25,  0]],

        [[36,  0, 81],
         [16,  9, 16]]])
"""

"""
# test numpy transpose functions
# np.moveaxis(a, sources, destinations)
# np.transpose(a, axes=None)
x = np.random.rand(3,2,2)
x = np.random.randint(0,10,(3,2,2))
y = np.moveaxis(x, 0, -1)   
y = np.transpose(x, [1,2,0])
z = torch.from_numpy(y)
t = torch.sum(z, [0,1])

x = np.random.randint(0,10,(3,2,2))
x
array([[[0, 5],
        [8, 5]],

       [[0, 1],
        [9, 0]],

       [[4, 8],
        [5, 0]]])
y = np.transpose(x, [1,2,0])
y
array([[[0, 0, 4],
        [5, 1, 8]],

       [[8, 9, 5],
        [5, 0, 0]]])
z = torch.from_numpy(y)
t = torch.sum(z, [0,1])
t
tensor([18, 10, 17])
"""

"""
# test numpy statistics
x = np.random.randint(0,10,(2,2,3))
x
array([[[0, 6, 8],
        [9, 4, 9]],

       [[1, 9, 9],
        [8, 3, 8]]])
y=torch.from_numpy(x)
ty = torch.sum(y, [0, 1])
ty
tensor([18, 22, 34])

z  = y**2
z
tensor([[[ 0, 36, 64],
         [81, 16, 81]],

        [[ 1, 81, 81],
         [64,  9, 64]]], dtype=torch.int32)
tz = torch.sum(z, [0,1])
tz
tensor([146, 142, 290])

npix = x.shape[0]*x.shape[1]
npix
4

mean = ty/npix
mean
tensor([4.5000, 5.5000, 8.5000])

r = tz - npix*mean**2
r
tensor([65., 21.,  1.])

std = (r/(npix-1))**0.5
std
tensor([4.6547, 2.6458, 0.5774])
"""


import json
import numpy as np
import copy
import cv2
import os
import sys
import torch

os.environ["LANDMARK_IMAGE_PATH"] = '../'
data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, '')        #'db', 'coco', 'images')

from os import listdir
from os.path import isfile, join

import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        my_imgfiles.append(onlyfiles[i])

pix = 1.0
sum_tot = torch.tensor([0, 0, 0])
sum_sq_tot = torch.tensor([0, 0, 0])
sum_pix = 0.0
for i in range(len(my_imgfiles)):
    img_name = os.path.join(image_root, my_imgfiles[i])  

    # NOTE: must convert to "int", otherwise image**2 will overflow 
    #       since results are limited to uint8 [0~255]
    # 
    image_cv = cv2.imread(img_name)
    image_cv = image_cv.astype(int)
    # np.sum(np.sum(image_cv<0, axis=0), axis=0)  --> count number of negative values
    # image = np.asarray(image_cv)                --> convert to numpy array if needed
    # image_cv = np.transpose(image_cv, 0, -1)    --> transpose the axis if needed
    
    image = torch.from_numpy(image_cv)
    if(3 != image.shape[2]):
        continue
    sum = torch.sum(image, dim=[0, 1])
    sumsq = torch.sum(image**2, dim=[0, 1])
    pix = image.shape[0]*image.shape[1] 

    # NOTE: you can test each image with the following codes (it should be: std1 == std2)
    # avg = sum/pix    
    # image_avg = torch.ones(image.shape) * avg
    # image_dif = image - image_avg
    # std1 = torch.sum(image_dif**2, [0, 1])/(pix - 1)
    # std2 = (sumsq - pix* avg**2)/(pix -1)
    
    sum_tot = sum_tot + sum
    sum_sq_tot = sum_sq_tot + sumsq
    sum_pix = sum_pix + pix
    print(i, '/', len(my_imgfiles))

mean = sum_tot/sum_pix
std = (sum_sq_tot - sum_pix*mean**2) / (sum_pix -1.0)
std = std**0.5

print(mean)
print(std)

