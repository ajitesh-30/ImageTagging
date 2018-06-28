# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:44:37 2018

@author: HP-USER
"""

import pandas as pd
import cv2 as cv
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

X_train = []
X_test=[]
root="G:/INTERNSHIP IIITA/Dataset/NUS-WIDE/Images/Flickr/Flickr/"

#============================Train Images======================================
file=open("G:/INTERNSHIP IIITA/Dataset/NUS-WIDE/ImageList/ImageList/TrainImagelist.txt","r")


for i in range(0,5):
    temp=file.readline().replace('\\','/')
    temp_path=(root+temp).strip('\n')
    image=cv.imread(temp_path)
    #cv.imshow("image",image)
    #cv.waitKey(0)
    print(image.shape)
    resized=cv.resize(image,(256,256),interpolation = cv.INTER_AREA)
    cv.imshow("image",resized)
    cv.waitKey(0)
    
    print(resized.shape)
    imgarr = img_to_array(resized, data_format=None)
    temp = cv.cvtColor(imgarr, cv.COLOR_BGR2RGB)
    X_train.append(temp)
print('done')



#====================================Test Images=================================
test_file=open("G:/INTERNSHIP IIITA/Dataset/NUS-WIDE/ImageList/ImageList/TestImagelist.txt","r")

for i in range(0,3):
    temp=test_file.readline().replace('\\','/')
    temp_path=(root+temp).strip('\n')
    image=cv.imread(temp_path)
    #cv.imshow("image",image)
    #cv.waitKey(0)
    print(image.shape)
    resized=cv.resize(image,(256,256),interpolation = cv.INTER_AREA)
    cv.imshow("image",resized)
    cv.waitKey(0)
    print(resized.shape)
    imgarr = img_to_array(resized, data_format=None)
    temp = cv.cvtColor(imgarr, cv.COLOR_BGR2RGB)
    X_test.append(temp)
        
print('done')


X_train = np.array(X_train)
X_test=np.array(X_test)


X_train=X_train.astype('float32')
X_train=X_train/255.0

X_test=X_test.astype('float32')
X_test=X_test/255.0

print("X_train shape is ",X_train.shape)
print("X_test shape is ",X_test.shape)


from matplotlib import pyplot as plt

plt.subplot(121)
plt.imshow(X_train[4])


plt.subplot(122)
plt.imshow(X_test[2])
plt.show()

#======================================Train tags=============================================

train_tag_file=open("G:/INTERNSHIP IIITA/Dataset/NUS-WIDE/Testing_Image/Train_Tags1k.txt","r")

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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

model = Sequential()
model.add(Conv2D(11,3,3,activation='relu',input_shape=(256,256,3)))
model.add(Conv2D(9,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(5,3,3,activation='relu'))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
epochs = 1
lrate = 0.002
decay = lrate/epochs
sgd =SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(X_train,y_temp,epochs=epochs,batch_size=32)
