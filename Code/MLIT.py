# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:23:10 2018

@author: ML.IITA
"""


import pandas as pd
import cv2 as cv
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

X_train = []
X_test=[]
root="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/TestingSampleImages/"


training_imgs_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/TrainImagelist.txt"
testing_imgs_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/TestImagelist.txt"


train_tags_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Train_Tags1k.txt"

all_tagslist_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/tags_list.csv"

train_img_tag_table_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/Train_tags_1k.csv"

#============================Train Images======================================
file=open(training_imgs_path,"r")


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
test_file=open(testing_imgs_path,"r")

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
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1000,activation='relu'))

epochs = 1
lrate = 0.002
decay = lrate/epochs
sgd =SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print(model.summary())
#model.fit(X_train,y_temp,epochs=epochs,batch_size=32)
#model.save("model1.0")
model.load_weights("model1.0")

check=model.predict_classes(X_test[0].reshape(1,256,256,3))
print(check)
check=model.predict_proba(X_test[0].reshape(1,256,256,3))

#==========================End of CNN Model=================

#==============Creating Matrices=================================================

#==================================Train : img_no X tags(One hot vector) Matrix==========================================
train_img_X_tag_table_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/Train_tags_1k.csv"
train_img_X_tag_table=pd.read_csv(train_img_X_tag_table_path)
print(train_img_X_tag_table)

#=================================Train : serial X  image ID (path) ============================================
train_serial_X_imgid_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/Train_imgids.csv"
train_serial_X_imgid=pd.read_csv(train_serial_X_imgid_path)
print(train_serial_X_imgid)

#=================================Test : img_no X tags(One hot vector) Matrix==========================
test_img_X_tag_table_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/Test_tags_1k.csv"
test_img_X_tag_table=pd.read_csv(test_img_X_tag_table_path)
print(test_img_X_tag_table)

#=================================Test : serial X  image ID (path)=============================================
test_serial_X_imgid_path="C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/Test_imgids.csv"
test_serial_X_imgid=pd.read_csv(test_serial_X_imgid_path)
print(test_serial_X_imgid)



#===Converting the result probabilities array to dict and Sorting the probabilities in decending order==============


keys=np.arange(0,1000)
check=check.reshape(1000,1)
print(check.shape)
h=np.array(check)
mydict = dict(zip(keys, h))

import operator
sorted_dict = sorted(mydict.items(), key=operator.itemgetter(1),reverse=True)

final_probs=np.array(sorted_dict)
print(final_probs[0])  #Prints the 1st tag and its probability after sorted in decending order




#=====================================Tags to Words===========================
all_tags=pd.read_csv(all_tagslist_path)
all_tags_list=np.array(all_tags) #Creating numpy array of all possible tags
print(all_tags_list[0])
#print(all_tags_list[725])

def print_tags(mylist):
    for i in mylist:
        print(all_tags_list[i])  #Prints Tags in words

print_tags([0,4,10,58,46])
image=cv.imread("C:/Users/ML.IITA/Desktop/Multi_Label_Image_Tagging/Testing_Data/sunset.png")
resized=cv.resize(image,(256,256),interpolation = cv.INTER_AREA)
cv.imshow("image",resized)
cv.waitKey(0)



#================For cross verifying the actuall tags of tested image===========
index=0 # Index of image in training set

tags_onehot=test_img_X_tag_table.iloc[index] # Reading one hot vector for corresponding index image
tags_onehot=tags_onehot.values.reshape(1,1000)
print(tags_onehot.shape)

print(np.count_nonzero(tags_onehot[0]==1)) # Counting how many tags are assigned for that image

tags_assigned=[index for index, value in enumerate(tags_onehot[0]) if value == 1] #Finding the indexes of all tags which are assigned
print(tags_assigned)

print_tags(tags_assigned)  #Calling above print tags function






