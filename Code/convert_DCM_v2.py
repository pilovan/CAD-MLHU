# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:31:55 2021

This program produces PNG images with various HU windowing settings. 
DICOM files only

@author: pilovan
"""

import pydicom
import numpy as np
import cv2
import os

##### Filepaths for saving and loading
filepath_loading = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted'
filepath_saving = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Repository B\Series_Windowing'

##### Window Settings : Abdomen/Chest, Angio, Bone, Brain, Lungs, Dynamic
window_center = [40, 300, 300, 40,-400, 176]
window_width = [400, 600, 1500, 80, 1500, 2400]

##### Only consider DICOM file incase another file exists
files = [file for file in os.listdir(filepath_loading) if ".dcm" in file]

for file in files:

    ##### read file and save to dataframe. Contains CT data & patient data
    df = pydicom.dcmread(filepath_loading + "/" + file)
    ##### withdraw CT data
    pix_array = df.pixel_array
    ##### Produce hounsfield array
    pix_array = pix_array * int(df.RescaleSlope) + int(df.RescaleIntercept)
    ##### find shape of original image and create new image with the same shape
    array_shape = pix_array.shape
    new_img = np.zeros((array_shape[0], array_shape[1]), np.uint8)
        
    ##### for each window settings, create new image 
    for i in range(len(window_center)): 
        
        ##### Set window max & min
        window_max = int(window_center[i] + window_width[i] / 2)
        window_min = int(window_center[i] - window_width[i] / 2)
        
        ##### create new image using window max & min
        for x in range(array_shape[0]):
            for y in range(array_shape[1]):
                if (pix_array[x][y] > window_max):
                    new_img[x][y] = 255
                elif (pix_array[x][y] < window_min):
                    new_img[x][y] = 0
                else:
                    new_img[x][y] = int(((pix_array[x][y] - window_min) / (window_max - window_min)) * 255)
                    
        ##### superimpose in rectangle in new image to remove some unwanted data
        cv2.rectangle(new_img, (512,400),(0, 512),(0,0,0), cv2.FILLED)

        ##### save new image 
        file_name_new = 'Rep_2_' + file[:-4] + '_WC_' + str(window_center[i]) + '_WW_' + str(window_width[i]) + '.png'
        file_name_new = filepath_saving + "/" + file_name_new   
        print(file_name_new)
        cv2.imwrite(file_name_new, new_img)
