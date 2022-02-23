# @MxTan from SpaceVision SZ Co.Ltd 
#
# @brief  for display landmark annotations piece by piece
#
# ref. windows version cocoapi if you need a mask version
# https://github.com/philferriere/cocoapi
#
#

from CoLandMark import LandMark
import numpy as np
import skimage.io as io #conda install scikit-image
import json
import os
 
import matplotlib as mpl
mpl.use('TkAgg')
import pylab
import matplotlib.rcsetup as rcsetup
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
 
dataDir='D:/deepData/widerface/WIDER_train/images'
annFile= 'libfacedetection_tools/trainset.json'
 
# initialize COCO api for instance annotations
coco=LandMark(annFile)

# display COCO categories
catIds = coco.getCatIds()
cats = coco.loadCats(catIds)
nms=[cat['name'] for cat in cats] 
print('COCO format categories: \n{}\n'.format(' '.join(nms)))
 
# recursively display all images and its masks
imgIds = coco.getImgIds()
for id in imgIds: 
    mpl.pyplot.clf()  #put a stop breakpoint here, each cycle you will see a marked image
    annIds = coco.getAnnIds([id], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    imgIds = coco.getImgIds(imgIds = [id])
    img = coco.loadImgs(imgIds[0])[0]

    #----- save seperate image ----
    #file_name_ext='./WIDER_train/images/' + img['file_name']
    #(filename,extension) = os.path.splitext(file_name_ext)
    #file_path = "coco/" + filename + ".json"
    #data = {"annotations":anns}
    #with open(file_path, 'w') as result_file:
    #    json.dump(data, result_file)
    
    #----display image----
    file_path = '{}/{}'.format(dataDir,img['file_name'])
    I = io.imread(file_path) 
    #NOTE: the above method is equivalent to the following format
    #I = io.imread('%s/%s'%(dataDir,img['file_name']))  

    mpl.pyplot.imshow(I)
    mpl.pyplot.axis('off')
    coco.showAnns(anns)
