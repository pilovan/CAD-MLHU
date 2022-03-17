# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:20:14 2021

@author: pilovan
"""
import os
import cv2
import numpy as np

filepath_loading = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Collages\To Combine'
filepath_saving = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Collages\Combined'
os.chdir(filepath_loading) #uint16bitimage
files = os.listdir(filepath_loading)

images = list()

for x in range(len(files)):
    image = cv2.imread(files[x])
    images.append(image)
    
collage = np.hstack((images[0], images[2], images[1]))

filename = filepath_saving + "/" + "Collage_" + files[0]
cv2.imwrite(filename,collage )