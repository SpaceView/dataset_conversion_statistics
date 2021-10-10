# Retinaface_annoation_format

Retinaface的标注每条共计有16个数据，

ref . <Yolov5face/data/ train2yolo.py>

```python
第0到13个
for idx, label in enumerate(labels):
            annotation = np.zeros((1, 14))
            # bbox
            label[0] = max(0, label[0])
            label[1] = max(0, label[1])
            label[2] = min(width -  1, label[2])
            label[3] = min(height - 1, label[3])
            annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
            annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
            annotation[0, 2] = label[2] / width  # w
            annotation[0, 3] = label[3] / height  # h
            #if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
            #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
            #    continue
            # landmarks
            annotation[0, 4] = label[4] / width  # l0_x
            annotation[0, 5] = label[5] / height  # l0_y
            annotation[0, 6] = label[7] / width  # l1_x
            annotation[0, 7] = label[8]  / height # l1_y
            annotation[0, 8] = label[10] / width  # l2_x
            annotation[0, 9] = label[11] / height  # l2_y
            annotation[0, 10] = label[13] / width  # l3_x
            annotation[0, 11] = label[14] / height  # l3_y
            annotation[0, 12] = label[16] / width  # l4_x
            annotation[0, 13] = label[17] / height  # l4_y
            str_label="0 "
            for i in range(len(annotation[0])):
                str_label =str_label+" "+str(annotation[0][i])
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            f.write(str_label)
```

另外，参考<[Annotation format for RetinaFace](https://github.com/deepinsight/insightface/issues/664)>

```
第14个：
0.0 refers to visible, 
1.0 refers to invisible,
0表示容易，1表示困难
第15个
and the last floating point refers to blur
1个置信度值

-1/0/1 denotes the state of unvalid/indisputable/annotatable
```

