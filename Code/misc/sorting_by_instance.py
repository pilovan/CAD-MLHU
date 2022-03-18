# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:11:12 2021

What: Produces new DCM files with names determined by instance.

Why: To organize the repositories and produce useful references.

INPUT: DCM FILES
OUTPUT: DCM FILES 

@author: pilovan
"""

import pydicom
import os

##### filepath for loading and saving images
filepath_loading  = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT'
filepath_saving = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_by_instance'

files = [file for file in os.listdir(filepath_loading) if ".dcm" in file]

for file in files:
    ##### load files and store in dataframe
    df = pydicom.dcmread(filepath_loading + "/" + file) 
    
    ##### determine new filename       
    filename = filepath_saving + '/' + 'CT_instance_' + str(df.InstanceNumber) + '.dcm'
    
    ##### save new files
    df.save_as(filename)
            