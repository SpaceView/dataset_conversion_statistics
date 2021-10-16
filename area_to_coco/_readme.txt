coco_to_seperate.py 
ref. 《保存coco dataset注释为单一文件，并逐一显示所有图片的mask》
https://blog.csdn.net/tanmx219/article/details/90726504
官方的例子只显示 一张图片，我需要逐一显示，并且官方的那个JSON文件太大了，我把注释文件分开存储，每张图片一个注释文件，另行保存在一个叫coco的文件夹中，


area_to_coco.py
由AutoSeg标注生成的area像素级标注文件，生成一个统一的coco文件格式，用于训练模型
