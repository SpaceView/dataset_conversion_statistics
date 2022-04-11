（一）安装labelme，并通过labelme进行标注。
标注图片尺寸目前是800x800, 当然可以是任意其他尺寸。
这样做的目的是可以使用任何尺寸的图片作为来源，后续统一处理后喂料给模型。例如，
800x800 ---> 640x640
800x600 ---> 640x640
其他尺寸经cut或padding后，无畸变缩放成640x640或640x480

（二）通过labelme标注后，首先通过labelme下面的labelme2coco_polygon.py将所有的图片转变为coco格式，该程序会生成：
（1）包括所有标注的JSON文件
（2）为每个图片生成一个JSON文件

（三）通过resize_image_and_annotation_std640x640.py将图片生成标准尺寸格式：
640x640，或640x480
说明：为了应用方便，目前所有训练统一采用宽度为640的图片：640x640，640x480。

（四）通过scaling_generalization进行缩放泛化

（五）通过generate_rot_imgs.py进行角度泛化

（六）各种生成的标注，可以通过coco2lableme再转换成lableme格式，以用labelme软件查看最终的转换结果是否正确。
https://github.com/SpaceView/labelme/blob/main/examples/instance_segmentation/coco2labelme_polygon.py


