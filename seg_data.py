import cv2
import matplotlib.pyplot as plt
import os
import shutil

VOC_dir = '/usr/src/Torch_test/data/VOC2012'
train_file = VOC_dir+'/ImageSets/Segmentation/train.txt'
val_file = VOC_dir +'/ImageSets/Segmentation/val.txt'

print(os.listdir(VOC_dir))

train_list = []
val_list = []

file = open(train_file,'r')
while True:
    line=file.readline().strip()
    #print(line)
    if line=='':
        break
    train_list.append(line)
file.close()

file = open(val_file,'r')
while True:
    line = file.readline().strip()
    if line=='':
        break
    val_list.append(line)
file.close()

# 파일 이동
To_path = '/usr/src/Torch_test/FCN/data'
from_path = '/usr/src/Torch_test/data/VOC2012/JPEGImages'
label_path = '/usr/src/Torch_test/data/VOC2012/SegmentationClass'

train_path= To_path+'/train'
val_path=To_path+'/val'
train_label=To_path+'/train_label'
val_label=To_path+'/val_label'

for f in train_list:
    f_train = from_path+'/'+f+'.jpg'
    shutil.copy2(f_train,train_path)
    label = label_path+'/'+f+'.png'
    shutil.copy2(label,train_label)
    

for f in val_list:
    f_val = from_path+'/'+f+'.jpg'
    shutil.copy2(f_val,val_path)
    label = label_path+'/'+f+'.png'
    shutil.copy2(label,val_label)

