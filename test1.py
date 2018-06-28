# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:14:13 2018

@author: HP-USER
"""

import os

root='G:/INTERNSHIP IIITA/Dataset/NUS-WIDE/Testing_Image/'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
print (dirlist)



import numpy as np
dirlist = np.sort(dirlist)

from keras.preprocessing.image import load_img, img_to_array
x_train = []
for dirs in range(len(dirlist)):
    for image_path in os.listdir(root+dirlist[dirs]):
        if not image_path.endswith('.db'):
            img = load_img(
        root+str(dirlist[dirs])+'/'+str(image_path),
        grayscale=False,
        target_size=(256,256),
        interpolation='nearest'
    )
            img = img_to_array(img, data_format=None)
            print(root+str(dirlist[dirs])+'/'+str(image_path))
            x_train.append(img)

print('done')


x_train = np.array(x_train)

print(x_train.shape)

x_train=x_train.astype('float32')
x_train=x_train/255.0

from matplotlib import pyplot as plt

plt.subplot(111)
plt.imshow(x_train[5].reshape(256,256,3))
plt.show()