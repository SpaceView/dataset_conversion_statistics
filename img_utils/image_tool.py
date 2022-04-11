import json
import numpy as np
import copy
import cv2
import math

from random import uniform, randint

#You MUST update these values according to current requirements
from img_utils.labelme_util_constants import MAX_CAT_ID
from img_utils.labelme_util_constants import TRAIN_IMG_WDITH as TARGET_WIDTH
from img_utils.labelme_util_constants import TRAIN_IMG_HEIGHT as TARGET_HEIGHT
from img_utils.labelme_util_constants import TRAIN_SCALE_LO as SCALE_LO
from img_utils.labelme_util_constants import TRAIN_SCALE_HI as SCALE_HI

from random import randint

PI = 3.1415926535897932
MASKCHAR = 255
EDGE_BUFFER = 10
REQUIRE_SCALING = True


def RotateImage(_src, angle):
    rad = angle / 180.0 * PI
    sinVal = abs(math.sin(rad))
    cosVal = abs(math.cos(rad))
    width = _src.shape[1]
    height = _src.shape[0]
    target_wd = (int)(width * cosVal + height * sinVal)
    target_ht = (int)(width * sinVal + height * cosVal)
    dx = math.ceil((target_wd - width)/2)
    dx = max(dx, 0) # ensure dx>=0
    dy = math.ceil((target_ht - height)/2)
    dy = max(dy, 0) # ensure dy>=0
    target_wd = dx*2 + width
    target_ht = dy*2 + height
    dx = max(dx,0)
    dy = max(dy,0)
    _dst = cv2.copyMakeBorder(_src, dy, dy, dx, dx, cv2.BORDER_CONSTANT)
    #assert(target_wd == _dst.shape[1])
    #assert(target_ht == _dst.shape[0])
    ptCenter = ( int(target_wd / 2), int(target_ht / 2 ))
    affine_matrix = cv2.getRotationMatrix2D(ptCenter, angle, 1.0)
    _dst = cv2.warpAffine(_dst, affine_matrix, dsize=(target_wd, target_ht), flags = cv2.INTER_NEAREST)
    return _dst, ptCenter


def CropImage(_src, rc):
    x0 = rc[0]
    x1 = rc[0] + rc[2]
    y0 = rc[1]
    y1 = rc[1] + rc[3]
    dst = _src[y0:y1, x0:x1]
    return dst



def shift_image_t(_src, xsht, ysht):
    xabs = abs(xsht)
    yabs = abs(ysht)
    ht = _src.shape[0]
    wd = _src.shape[1]
    dst = np.zeros(_src.shape, np.uint8)
    # first, we copy from _src to dst with x transition
    if xsht < 0:
        dst[0:ht, 0:(wd-xabs)] = _src[0:ht, xabs:wd]
        dst[0:ht, (wd-xabs):wd] = _src[0:ht, 0:xabs]
    elif xsht>0:         
        dst[0:ht, 0:xabs] = _src[0:ht, (wd-xabs):wd]
        dst[0:ht, xabs:wd] = _src[0:ht, 0:(wd-xabs)]
    # then, we copy from dst to _src with y transition
    if ysht < 0:
        _src[0:(ht-yabs), 0:wd] = dst[yabs:ht, 0:wd]
        _src[(ht-yabs):ht, 0:wd] = dst[0:yabs, 0:wd]
    elif ysht>0: 
        _src[0:yabs, 0:wd] = dst[(ht-yabs):ht, 0:wd]
        _src[yabs:ht, 0:wd] = dst[0:(ht-yabs), 0:wd]        
    return _src


def shift_polygon_t(points, xsht, ysht):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)==len(ypts))
    dxs = xpts + xsht
    dys = ypts + ysht
    arr = np.stack((dxs, dys), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return dxs, dys, arr


def pad_image_t(_src, _width, _height):
    ht = _src.shape[0]
    wd = _src.shape[1]
    assert(_height >= ht)
    assert(_width >= wd)
    t_top = int((_height - ht)/2)
    t_bot = _height - ht - t_top
    t_left = int((_width - wd)/2)
    t_right = _width - wd - t_left
    t_img = cv2.copyMakeBorder(src=_src, top=t_top, bottom=t_bot, left=t_left, right=t_right,
                                borderType=cv2.BORDER_CONSTANT,value=[114, 114, 114])
    return t_img, (t_left, t_top, t_right, t_bot)


def extract_bbox_t(points):
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    bbox = (xpts.min(),ypts.min(),xpts.max(),ypts.max())
    return bbox


def extract_ann_bbox_t(shapes, _width, _height):
    rcbb = [_width, _height,0,0]
    for shape in  shapes:
        box = extract_bbox_t(shape['points'])
        rcbb[0] = min(box[0], rcbb[0]) #left
        rcbb[1] = min(box[1], rcbb[1]) #top
        rcbb[2] = max(box[2], rcbb[2]) #right
        rcbb[3] = max(box[3], rcbb[3]) #bottom
    rcbb[0] = max(0, rcbb[0])
    rcbb[1] = max(0, rcbb[1])
    rcbb[2] = min(_width, rcbb[2])
    rcbb[3] = min(_height, rcbb[3])
    return rcbb


def scale_image_at_random_scale_t(_src, _annbbox, _tgt_wd, _tgt_ht):
    #(1) resize image at some random scale
    wd = _src.shape[1]
    ht = _src.shape[0]
    wdr = _annbbox[2] - _annbbox[0] + 2*EDGE_BUFFER  # we need some edge for buffering
    htr = _annbbox[3] - _annbbox[1] + 2*EDGE_BUFFER
    xrup = min(float(wd)/wdr, SCALE_HI)
    yrup = min(float(ht)/htr, SCALE_HI)
    #xdir = randint(0,1)
    #ydir = randint(0,1)
    #if(xdir):  # 1 --> enlarge
    #    xscale = uniform(1.0, xrup)
    #else:      # 0 --> reduce
    #    xscale = uniform(0.75, 1.0)
    #if(ydir):
    #    yscale = uniform(1.0, yrup)
    #else:
    #    yscale = uniform(0.75, 1.0)        
    rup = min(xrup, yrup)
    dir = randint(0,1)
    if(dir):
        scale = uniform(1.0, rup)
    else:
        scale = uniform(SCALE_LO, 1.0)
    xscale = scale
    yscale = scale            
    dest_ht = int(ht * yscale)
    dest_wd = int(wd * xscale)
    dim = (dest_wd, dest_ht)
    img_rsz = cv2.resize(_src, dim, interpolation=cv2.INTER_AREA)
    anbox_rsz = [_annbbox[0]*xscale, _annbbox[1]*yscale, _annbbox[2]*xscale, _annbbox[3]*yscale]
    # (2) pad the image ensure (width > target_width) and (height > target_height)
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0
    if(dest_ht < _tgt_ht):
        pad_top = int((_tgt_ht - dest_ht)/2)
        pad_bot = _tgt_ht - dest_ht - pad_top
    if(dest_wd < _tgt_wd):
        pad_left = int((_tgt_wd - dest_wd)/2)
        pad_right = _tgt_wd - dest_wd - pad_left
    t_img = cv2.copyMakeBorder(src=img_rsz, top=pad_top, bottom=pad_bot, left=pad_left, right=pad_right,
                                borderType=cv2.BORDER_CONSTANT,value=[114, 114, 114])
    # (3) Crop the annotation area to target dimension
    xcenter = int((anbox_rsz[0]+anbox_rsz[2])/2)
    ycenter = int((anbox_rsz[1]+anbox_rsz[3])/2)
    crop_x0 = min(int(xcenter - _tgt_wd/2), int(t_img.shape[1] - _tgt_wd))
    crop_y0 = min(int(ycenter - _tgt_ht/2), int(t_img.shape[0] - _tgt_ht))
    crop_x0 = max(0, crop_x0)
    crop_y0 = max(0, crop_y0)
    crop_x1 = crop_x0 + _tgt_wd
    crop_y1 = crop_y0 + _tgt_ht
    img_crop = t_img[crop_y0:crop_y1, crop_x0:crop_x1]    
    scale_pad_cut_info = ([xscale, yscale], [pad_left, pad_top], [crop_x0, crop_y0])
    if(img_crop.shape[0]!=640):
        print('error ht')
    if(img_crop.shape[1]!=640):
        print('error wd')
    return img_crop, scale_pad_cut_info

def scale_polygon_t(points, scale_info):
    scl = scale_info[0]  #[xscale, yscale]
    pad = scale_info[1]  #[t_left, t_top] for pad
    cut = scale_info[2]  #[x0, y0] for cut
    pts = points
    if not isinstance(pts, np.ndarray):        
        pts = np.array(pts)
    l_pts = len(pts)
    xpts = pts[0:l_pts, 0]
    ypts = pts[0:l_pts, 1]
    assert(len(xpts)==len(ypts))
    dxs = xpts*scl[0] + pad[0] - cut[0]
    dys = ypts*scl[1] + pad[1] - cut[1]
    arr = np.stack((dxs, dys), axis = -1) #np.vstack((dxs, dys))   #arr = arr.flatten('F')
    return dxs, dys, arr
