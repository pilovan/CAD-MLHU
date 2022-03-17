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

class labels():
    
    proceed = True
        
    def __init__(self, filepath_loading, filepath_saving, material, ifmask):
        
        self.filepath_loading = filepath_loading
        self.filepath_saving = filepath_saving
        self.material = material.lower()
        self.ifmask = ifmask
        
        self.determineInequality()
        if self.proceed == True:
            self.produceLabel()
        
    def determineInequality(self):
    ##### determines the HU ranges values for the chosen material
    
        if self.material == "bone":
            self.lower = 250
            self.upper = 3000
    
        elif self.material == "muscle":
            self.lower = 10
            self.upper = 40
        
        elif self.material == "water":
            self.lower = -5
            self.upper = 5
        
        elif self.material == "fat":
            self.lower = -100
            self.upper = -50
        
        elif self.material == "lungs":
            self.lower = -900
            self.upper = -500
            
        elif self.material == "air":
            self.lower = -1000
            self.upper = -900
            
        else:
            self.proceed = False 
            print("Please recheck spelling, or no such material")
            
                
    def produceLabel(self):
        
        for file in os.listdir(self.filepath_loading)[:4]:
            print(file)
            ##### read the DICOM image
            dataset = pydicom.dcmread(filepath_loading + "/" + file)
            
            ##### convert the scan data to HU data
            hu_data = dataset.pixel_array * int(dataset.RescaleSlope) + int(dataset.RescaleIntercept)
            
            ##### create a new image using the HU data. The new image will be the "label" in the ML model.
            new_image = np.zeros((512,512), dtype=np.uint8)
            new_image[(hu_data > self.lower) & (hu_data < self.upper)] = 255 
                                                  
            ##### Optional: superimpose mask 
            if (ifmask == True):
                self.produceMask()
                new_image = cv2.bitwise_or(new_image, new_image, mask = self.mask_inverse)
            
            ##### show label using cv2
            # cv2.imshow("Label", new_image)
            
            ##### save the new image as PNG
            filename = file[:-4] + "_" +  self.material + "_label" + ".png"
            filename = filepath_saving + '/' + filename
            cv2.imwrite(filename, new_image)
            
            ##### must exist for CV to hold window open and close window, for each image.
            cv2.waitKey(0)
            cv2.destroyAllWindows
            
    def produceMask(self):            
        #### Optional: superimpose masks to remove non-wanted data
        arr_overlay = np.zeros((512,512), dtype=np.uint8)
        cv2.circle(arr_overlay, center = (185,290), radius =45, color =(20,100,50), thickness=-1)
        cv2.circle(arr_overlay, center = (300,290), radius =45, color =(20,100,50), thickness=-1)
        cv2.circle(arr_overlay, center = (275,320), radius =10, color =(20,100,50), thickness=-1)
        cv2.rectangle(arr_overlay, (200,240),(300, 300),(20,100,50), cv2.FILLED)
        ret, mask = cv2.threshold(arr_overlay, 0, 255, cv2.THRESH_BINARY)
        self.mask_inverse = cv2.bitwise_not(mask)

            
if __name__ == '__main__':
    
    ##### filepaths for loading DCM files and saving PNG files
    filepath_loading = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted'
    filepath_saving = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\Masks\Masks_V2'

    ##### type of materials: bone, muscle, water, fat, lungs, air
    material = "bone"
    
    ##### mask = True or False
    ifmask = False
    
    labels(filepath_loading, filepath_saving, material, ifmask)
    
