//=================================================
//《area_to_coco》
//=================================================
area_to_coco.py    ----> organized AutoSegProfessional annotated items into a  combined coco file
coco_disp_bbox.py  ----> display coco dataset with BBOX，to verify the correctness of your bbox
coco_area_to_seperate.py  ----> to verify correctness of your coco dataset format
coco_to_seperate.py ----> make seperate coco annotations for each image file, 
                                       since the original coco annotation file is too big.

//=================================================
//《image_mean_std》
//=================================================
You can use these codes to extract mean and std for your own dataset



//=================================================
//《landmark_to_yolo》
//=================================================
retina_face format 
"""
# xxxx.jpg
bbx bby bbw bbh lmx1 lmy1 lmt1 lmx2 lmy2 lmt2 lmx3 lmy3 lmt3 lmx4 lmy4 lmt4 lmx5 lmy5 lmt5
bbx bby bbw bbh lmx1 lmy1 lmt1 lmx2 lmy2 lmt2 lmx3 lmy3 lmt3 lmx4 lmy4 lmt4 lmx5 lmy5 lmt5
# yyyy.jpg
bbx bby bbw bbh lmx1 lmy1 lmt1 lmx2 lmy2 lmt2 lmx3 lmy3 lmt3 lmx4 lmy4 lmt4 lmx5 lmy5 lmt5
bbx bby bbw bbh lmx1 lmy1 lmt1 lmx2 lmy2 lmt2 lmx3 lmy3 lmt3 lmx4 lmy4 lmt4 lmx5 lmy5 lmt5
bbx bby bbw bbh lmx1 lmy1 lmt1 lmx2 lmy2 lmt2 lmx3 lmy3 lmt3 lmx4 lmy4 lmt4 lmx5 lmy5 lmt5
"""
this format is used for the following models,
retinaface
yolov5face
dbface

NOTE: there might be minor differences for different models.

//=================================================
//《landmark_to_coco》
//=================================================
coco landmark format

this format is used for the following models,
libfacedetection