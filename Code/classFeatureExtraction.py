# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:50:52 2018

@author: HULK
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import hog
from skimage import data ,exposure
from PIL import Image


def siftExtractor(imgPath):
    img = cv2.imread(imgPath)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift= cv2.xfeatures2d.SIFT_create(400)
    sift.setHessianThreshold(1000)
    sift.setExtended(1)
    kp, des = sift.detectAndCompute(gray,None)
    return kp,des
    
def hogExtractor(imgPath):
    temp=np.asarray(Image.open(imgPath))
    x=temp.shape[0]
    y=temp.shape[1]*temp.shape[2]
    temp.resize((x,y)) 
    image=temp
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    return fd, hog_image

def surfExtractor(imgPath):
    img = cv2.imread(imgPath)
    surf =cv2.xfeatures2d.SURF_create(400)
    surf.setHessianThreshold(1000)
    surf.setExtended(1)
    kp, des = surf.detectAndCompute(img,None)
    return kp,des
        