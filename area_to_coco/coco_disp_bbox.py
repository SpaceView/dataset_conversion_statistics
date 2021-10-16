
from pycocotools.coco import COCO
import cv2
import pandas as pd
 
dataset_coco = 1

data_root = ''
data_type = ''
image_root = ''
annFile = ''

if(dataset_coco):
    data_root='E:/BigData/coco/VOC2017'
    data_type = 'val2017'
    image_root = '%s/%s'%(data_root,data_type)
    annFile='{}/annotations/instances_{}.json'.format(data_root, data_type)
else:
    data_root='E:/img/R0805_area/Rotated'
    data_type = 'R0805'
    image_root = '%s'%(data_root)
    annFile='{}/annotations/instances_{}.json'.format(data_root, data_type)

#NOTE: you MUST create the 'bboxsav' folder manually
bboxsav_root = data_root + '/bboxsav/'

#NOTE: these are image sequence ids whose bbox is to be displayed
image_seqs = [1,2,3]                      

def showNimages(annFile, imageFile):
    #You can use PANDA to read a list
    #data = pd.read_csv(imageidFile)  # csv file
    #list = data.values.tolist()    
    #for i in range(len(list)):
    #    image_id.append(list[i][0])

    print(image_seqs)
    print(len(image_seqs))
    coco = COCO(annFile)

    image_ids = []
    if(dataset_coco):
        imgIds = coco.getImgIds()
        image_ids = [imgIds[id] for id in image_seqs]
    else:
        image_ids = image_seqs

    for i in range(len(image_ids)):
        img = coco.loadImgs(image_ids[i])[0]
        image = cv2.imread('%s/%s'%(image_root, img['file_name']))
        annIds = coco.getAnnIds(imgIds=image_ids[i], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for n in range(len(anns)):
            x, y, w, h = anns[n]['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imwrite(bboxsav_root + str(image_ids[i]) + 'result.png', image)
    print("Images are generated at {}".format(bboxsav_root))


if __name__ == "__main__":	
    showNimages(annFile=annFile, imageFile=data_root)
    print('Done')