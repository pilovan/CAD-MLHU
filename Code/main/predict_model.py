# -*- coding: utf-8 -*-
"""

@author: Paul Ilovan 

This file initiates a model prediction/classification.

"""

import numpy as np
import cv2
import pandas as pd
import itertools
import pickle
import os

def createGaborKernel(cartesian_list):
    s = cartesian_list[0]
    t = cartesian_list[1]
    l = cartesian_list[2]
    g = cartesian_list[3]
    
    return cv2.getGaborKernel((9, 9), s, t, l, g, 0, ktype=cv2.CV_32F)    

def feature_extraction(img):
    df = pd.DataFrame()


    #All features generated must match the way features are generated for TRAINING.
    #Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Pixel_Values'] = img2

    from skimage.filters import roberts, sobel, scharr, prewitt

    #Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny'] = edges1 #Add column to original dataframe

    #Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    #Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1


    from scipy import ndimage as nd
    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median_s3'] = median_img1
    

    #Feature 8 is Gaussian with sigma
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian_s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian_s7'] = gaussian_img3

    variance = nd.generic_filter(img, np.var, size=3).reshape(-1) 
    df["Variance_s3"] = variance 
    
    theta = [1,2]
    sigma = [1,3]
    lamda = list(np.arange(0, np.pi, np.pi / 4))
    gamma = [0.05, 0.5]
    gabor_vars = list(itertools.product(*[sigma,theta,lamda,gamma]))
    gabor_kernel = list(map(createGaborKernel, gabor_vars))
    num = 1
    for kernel in gabor_kernel:
        Gabor = "Gabor_" + str(num)
        df[Gabor] = cv2.filter2D(np.float32(img), cv2.CV_8UC3, kernel).reshape(-1)
        num += 1
            
    # Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    # print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
                    # print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    # Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  
                    num += 1

    return df


#########################################################

#Applying trained model to segment multiple files. 



segmentation_dir = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Segmentation Model'
filename = segmentation_dir + "\bone_segmentation_model_5"
loaded_model = pickle.load(open(filename, 'rb'))

image_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\ct_all_png'
os.chdir(image_path)
files = os.listdir(image_path)

for image_test in files[:2]:
    print(image_test)
           
    img1= cv2.imread(image_test)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    #Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)

    segmented = result.reshape((img.shape))
    segmented = segmented.astype(np.uint8)
    
    img = cv2.merge([img,img,img])
    
    segmented_layered = cv2.merge([segmented,segmented, segmented])
    segmented_layered[segmented_layered[:,:, 1] > 0] = [0,255,0]

    rows, cols, channels = segmented_layered.shape
    roi = img[0:rows, 0:cols]
    
    ret, mask = cv2.threshold(segmented, 10 ,255, cv2.THRESH_BINARY)
    mask_inverse = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inverse)
    img2_fg = cv2.bitwise_and(segmented_layered, segmented_layered,mask = mask )
    
    out_img = cv2.add(img1_bg, img2_fg)
    img[0:rows, 0:cols] = out_img
    
    image_test_2 = image_test[:-4] +  '_result_segmentation_' + filename + '_' + '.jpg'
    cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\Repository A Results' + "/" + image_test_2, img)
