# -*- coding: utf-8 -*-
"""

@author: Paul Ilovan 

This file creates the label data for the ML model

"""
 
import numpy as np
import cv2
import pandas as pd
import os
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import itertools

def reshape2D(pixels):
    return pixels.reshape(512,512)

def imageDataframe(file):
    image_temp = cv2.imread(file)
    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
    return image_temp.reshape(-1) 

def applyRoberts(image):
    return roberts(image).reshape(-1)

def applyCanny(image):
    return cv2.Canny(image, 100,200).reshape(-1)

def applySobel(image):
    return sobel(image).reshape(-1)

def applyScharr(image):
    return scharr(image).reshape(-1)
    
def applyPrewitt(image):
    return prewitt(image).reshape(-1)

def applyMedian(image):
    return nd.median_filter(image, size=3).reshape(-1)
    
def applyGaussianS3(image):
    return nd.gaussian_filter(image, sigma=3).reshape(-1)

def applyGaussianS7(image):
    return nd.gaussian_filter(image, sigma=7).reshape(-1)

def applyVarianceS3(image):
    return nd.generic_filter(image, np.var, size=3).reshape(-1)   

def createGaborKernel(cartesian_list):
    s = cartesian_list[0]
    t = cartesian_list[1]
    l = cartesian_list[2]
    g = cartesian_list[3]
    
    
    return cv2.getGaborKernel((9, 9), s, t, l, g, 0, ktype=cv2.CV_32F)    
 
def selectWindowing(file):
    if "_300_600" in file:
        return file
        
if __name__ == '__main__':
    
    #########################################################
    # DATA ACUISITION SECTION
    print("Began D.A.S")
    
    image_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_Sorted_WC_WW_PNG'
    os.chdir(image_path)
    df = pd.DataFrame()
    files = os.listdir(image_path)
    files = list(filter(selectWindowing, files))
   
    print(files)
    # create features
    pixel_values = list(map(imageDataframe, files))
    pixel_values_reshaped = list(map(reshape2D, pixel_values))
    roberts_filter = list(map(applyRoberts, pixel_values_reshaped))   
    canny_filter = list(map(applyCanny, pixel_values_reshaped))
    sobel_filter = list(map(applySobel, pixel_values_reshaped))
    scharr_filter = list(map(applyScharr, pixel_values_reshaped))
    prewitt_filter = list(map(applyPrewitt, pixel_values_reshaped))
    median_filter_s3 = list(map(applyMedian, pixel_values_reshaped))
    gaussian_filter_s3 = list(map(applyGaussianS3, pixel_values_reshaped))
    gaussian_filter_s7 = list(map(applyGaussianS7, pixel_values_reshaped))
    variance_filter_s3 = list(map(applyVarianceS3, pixel_values_reshaped))


    # flatten lists 
    pixel_values = list(itertools.chain(*pixel_values))
    roberts_filter = list(itertools.chain(*roberts_filter))
    canny_filter = list(itertools.chain(*canny_filter))
    sobel_filter = list(itertools.chain(*sobel_filter))
    scharr_filter = list(itertools.chain(*scharr_filter))
    prewitt_filter = list(itertools.chain(*prewitt_filter))
    median_filter_s3 = list(itertools.chain(*median_filter_s3))
    gaussian_filter_s3 = list(itertools.chain(*gaussian_filter_s3))
    gaussian_filter_s7 = list(itertools.chain(*gaussian_filter_s7))
    variance_filter_s3 = list(itertools.chain(*variance_filter_s3))
    
       
    # add features to dataframe
    # df['Image_Name'] = files
    df['Pixel_Values'] = pixel_values
    df['Roberts'] = roberts_filter
    df['Canny'] = canny_filter
    df['Sobel'] = sobel_filter
    df['Scharr'] = scharr_filter
    df['Prewitt'] = prewitt_filter
    df['Median_s3'] = median_filter_s3
    df['Gaussian_s3'] = gaussian_filter_s3
    df['Gaussian_s7'] = gaussian_filter_s7
    df['Variance_s3'] = variance_filter_s3   
    
 
    #gabor filter
    theta = [1,2]
    sigma = [1,3]
    lamda = list(np.arange(0, np.pi, np.pi / 4))
    gamma = [0.05, 0.5]
    gabor_vars = list(itertools.product(*[sigma,theta,lamda,gamma]))
    gabor_kernel = list(map(createGaborKernel, gabor_vars))
    num = 1
    for kernel in gabor_kernel:
        Gabor = "Gabor_" + str(num)
        df[Gabor] = cv2.filter2D(np.float32(pixel_values_reshaped), cv2.CV_8UC3, kernel).reshape(-1)
        num += 1
            
    # create masks
    mask_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\Masks\Masks_V2'  
    os.chdir(mask_path) 
    masks = os.listdir(mask_path)

    df2 = pd.DataFrame()
    label_value = list(map(imageDataframe, masks))
    label_value = list(itertools.chain(*label_value))
    df2['Label_Value'] = label_value
    
    print("Completed D.A.S")
 
    #########################################################
    # DATA FORMATTING SECTION
    
    print("Began D.F.S")
    
    dataset = pd.concat([df, df2], axis =1)
    
    os.chdir(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Bone_Classification_Data')
    dataset.to_pickle("Bone_Classification_Dataset.pkl")
