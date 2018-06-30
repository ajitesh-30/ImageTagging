# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:58:06 2018

@author: HULK
"""

import cv2 as cv
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from classFeatureExtraction import siftExtractor,surfExtractor,hogExtractor
from skimage.io import imread
from skimage.feature import hog
from skimage import data ,exposure
from PIL import Image


X_train = []
X_test=[]
root="C:/Users/srika/MLIT_Test_Data/TestingSampleImages/"


training_imgs_path="C:/Users/srika/MLIT_Test_Data/TrainImagelist.txt"
testing_imgs_path="C:/Users/srika/MLIT_Test_Data/TestImagelist.txt"
train_tags_path="C:/Users/srika/MLIT_Test_Data/Testing_Data/Train_Tags1k.txt"
all_tagslist_path="C:/Users/srika/MLIT_Test_Data/tags_list.csv"
train_img_tag_table_path="C:/Users/srika/MLIT_Test_Data/Testing_Data/Train_tags_1k.csv"


file=open(training_imgs_path,"r")
siftDescriptor=[]
surfDescriptor=[]
matchList=[]
for i in range(0,5):
    temp=file.readline().replace('\\','/')
    temp_path=(root+temp).strip('\n')
    kp,des=siftExtractor(temp_path)
    siftDescriptor.append(des)
print('done')



train_tag_file=open(train_tags_path,"r")

y_temp=None
for i in range(0,5):
    temp=train_tag_file.readline().split()
    temp=list(map(int,temp))
    temp=np.array([temp])
    if i==0:
        y_temp=temp
    else:
        y_temp=np.concatenate((y_temp,temp))

print(y_temp.shape)
count=len(y_temp)

myTemp=[x for x in range(0,count)]

print(myTemp)
bf = cv.BFMatcher()




for sd in siftDescriptor:
    test_path="test.jpg"
    testKP,testDES=siftExtractor(test_path)
    matches = bf.knnMatch(sd,testDES, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    matchList.append(len(good))


print(matchList)    
finalList=[g for _,g in sorted(zip(matchList,myTemp),reverse=True)]  
print(finalList)  
    
    

