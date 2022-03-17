# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:21:07 2021

#converts the dicom file to HU and saves the masks as a seperate image

@author: pilovan
"""


import pydicom
import numpy as np
import cv2
import os

# Creating masks to remove non-wanted data
# arr_overlay = np.zeros((512,512), dtype=np.uint8)
# cv2.circle(arr_overlay, center = (185,290), radius =45, color =(20,100,50), thickness=-1)
# cv2.circle(arr_overlay, center = (300,290), radius =45, color =(20,100,50), thickness=-1)
# cv2.circle(arr_overlay, center = (275,320), radius =10, color =(20,100,50), thickness=-1)
# cv2.rectangle(arr_overlay, (200,240),(300, 300),(20,100,50), cv2.FILLED)
# ret, mask = cv2.threshold(arr_overlay, 0, 255, cv2.THRESH_BINARY)
# mask_inverse = cv2.bitwise_not(mask)

class convertToHU():
    
    def __init__(self, filepath_loading, filepath_saving, material):
        self.material = material.lower()
        self.filepath_loading = filepath_loading
        self.filepath_saving = filepath_saving
        self.determineInequality()
        self.convert()
        
    def determineInequality(self):
    ##### determines the HU ranges values for the chosen material
    
        if self.material == "bone":
            self.lower = 250
            self.upper = 3000
        
        if self.material == "muscle":
            self.lower = 10
            self.upper = 40
        
        if self.material == "water":
            self.lower = -5
            self.upper = 5
        
        if self.material == "fat":
            self.lower = -100
            self.upper = -50
        
        if self.material == "lungs":
            self.lower = -900
            self.upper = -500
            
        if self.material == "air":
            self.lower = -1000
            self.upper = -900
                
    def convert(self):
        
        for file in os.listdir(self.filepath_loading)[:1]:
            ##### read the DICOM image
            dataset = pydicom.dcmread(filepath_loading + "/" + file)
            
            ##### convert the scan data to HU data
            hu_data = dataset.pixel_array * int(dataset.RescaleSlope) + int(dataset.RescaleIntercept)
            
            ##### create a new image using the HU data. The new image will be the "label" in the ML model.
            new_image = np.zeros((512,512), dtype=np.uint8)
            new_image[(hu_data > self.lower) & (hu_data < self.upper)] = 255 
                                                  
            ##### if we have a mask, use bitwise operator to combine images. 
            # bg = cv2.bitwise_or(arr_segment, arr_segment, mask = mask_inverse)
            
            ##### show label using cv2
            # cv2.imshow("HU Label", label)
            
            ##### save as png and save to directory
            filename = file[:-4] + "_bone_mask" + ".png"
            filename = filepath_saving + '/' + filename
            
            cv2.imwrite(filename, new_image)
            
            
if __name__ == '__main__':
    
    filepath_loading = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted'
    filepath_saving = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\Masks\Masks_V2'
    material = "bone"
    convertToHU(filepath_loading, filepath_saving, material)
    
    ##### must exist for CV to hold window open and close window. Can be placed anywhere in the code
    cv2.waitKey(0)
    cv2.destroyAllWindows