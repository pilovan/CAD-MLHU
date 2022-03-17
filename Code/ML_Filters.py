# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:32:00 2021

Must have same amount of images and masks to work properally

@author: pilovan
"""
 
import numpy as np
import cv2
import pandas as pd
import os
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import itertools
import matplotlib.pyplot as plt

def reshape2D(pixels):
    return pixels.reshape(512,512)

def imageDataframe(file):
    image_temp = cv2.imread(file)
    image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
    return image_temp

def applyRoberts(image):
    # return roberts(image).reshape(-1)
    return roberts(image)


def applyCanny(image):
    return cv2.Canny(image, 100,200)

def applySobel(image):
    return sobel(image)

def applyScharr(image):
    return scharr(image)
    
def applyPrewitt(image):
    return prewitt(image)

def applyMedian(image):
    return nd.median_filter(image, size=3)
    
def applyGaussianS3(image):
    return nd.gaussian_filter(image, sigma=3)

def applyGaussianS7(image):
    return nd.gaussian_filter(image, sigma=7)

def applyVarianceS3(image):
    return nd.generic_filter(image, np.var, size=3)  

def createGaborKernel(cartesian_list):
    s = cartesian_list[0]
    t = cartesian_list[1]
    l = cartesian_list[2]
    g = cartesian_list[3]
    
    
    return cv2.getGaborKernel((9, 9), s, t, l, g, 0, ktype=cv2.CV_32F)    
 
def applyGabor(kernel, image):
    return ["okay"]
    # return cv2.filter2D(np.float32(image), cv2.CV_8UC3, kernel).reshape(-1)

# def applyGabor2(image):
#     return cv2.filter2D(np.float32(image), cv2.CV_8UC3, gabor_kernel).reshape(-1)

def selectWindowing(file):
    if "_300_600" in file and "181" in file:
        return file


        
if __name__ == '__main__':
    
    #########################################################
    # DATA ACUISITION SECTION
    print("Began D.A.S")
    
    image_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\ct_all_png'
    os.chdir(image_path)
    df = pd.DataFrame()
    files = os.listdir(image_path)
    files = list(filter(selectWindowing, files))
   
    print(files)
    # create features
    pixel_values = list(map(imageDataframe, files))
    pixel_values_reshaped = list(map(reshape2D, pixel_values))
    # roberts_filter = list(map(applyRoberts, pixel_values_reshaped))   
    # canny_filter = list(map(applyCanny, pixel_values_reshaped))
    # sobel_filter = list(map(applySobel, pixel_values_reshaped))
    # scharr_filter = list(map(applyScharr, pixel_values_reshaped))
    # prewitt_filter = list(map(applyPrewitt, pixel_values_reshaped))
    # median_filter_s3 = list(map(applyMedian, pixel_values_reshaped))
    # gaussian_filter_s3 = list(map(applyGaussianS3, pixel_values_reshaped))
    # gaussian_filter_s7 = list(map(applyGaussianS7, pixel_values_reshaped))
    # variance_filter_s3 = list(map(applyVarianceS3, pixel_values_reshaped))
    #gabor filter
    theta = [1,2]
    sigma = [1,3]
    lamda = list(np.arange(0, np.pi, np.pi / 4))
    gamma = [0.05, 0.5]
    gabor_vars = list(itertools.product(*[sigma,theta,lamda,gamma]))
    gabor_kernel = list(map(createGaborKernel, gabor_vars))
    num = 1
    for kernel in gabor_kernel[11:12]:

        
        Gabor = "Gabor_" + str(num)
        gabor = cv2.filter2D(np.float32(pixel_values_reshaped), cv2.CV_8UC3, kernel)
        
        cv2.imshow("fff", gabor[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows 

        cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'gabor_10.jpg', (gabor[0]))

        num += 1
    # cv2.imshow("fff", roberts_filter[0])
    
  
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'Roberts_Filter.jpg', (roberts_filter[0]*255))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'Canny_Filter.jpg', (canny_filter[0]))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'Sobel_Filter.jpg', (sobel_filter[0]*255))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'Canny_Filter.jpg', (canny_filter[0]))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'scharr_filter.jpg', (scharr_filter[0]*255))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'prewitt_filter.jpg', (prewitt_filter[0]*255))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'median_filter_s3.jpg', (median_filter_s3[0]))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'gaussian_filter_s3.jpg', (gaussian_filter_s3[0]))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'gaussian_filter_s7.jpg', (gaussian_filter_s7[0]))
    # cv2.imwrite(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\BS_ML_5_Results\All_Filters' + "/" + 'variance_filter_s3.jpg', (variance_filter_s3[0]))

    
    # # flatten lists 
    # pixel_values = list(itertools.chain(*pixel_values))
    # roberts_filter = list(itertools.chain(*roberts_filter))
    # canny_filter = list(itertools.chain(*canny_filter))
    # sobel_filter = list(itertools.chain(*sobel_filter))
    # scharr_filter = list(itertools.chain(*scharr_filter))
    # prewitt_filter = list(itertools.chain(*prewitt_filter))
    # median_filter_s3 = list(itertools.chain(*median_filter_s3))
    # gaussian_filter_s3 = list(itertools.chain(*gaussian_filter_s3))
    # gaussian_filter_s7 = list(itertools.chain(*gaussian_filter_s7))
    # variance_filter_s3 = list(itertools.chain(*variance_filter_s3))
    
       
    # # add features to dataframe
    # # df['Image_Name'] = files
    # df['Pixel_Values'] = pixel_values
    # df['Roberts'] = roberts_filter
    # df['Canny'] = canny_filter
    # df['Sobel'] = sobel_filter
    # df['Scharr'] = scharr_filter
    # df['Prewitt'] = prewitt_filter
    # df['Median_s3'] = median_filter_s3
    # df['Gaussian_s3'] = gaussian_filter_s3
    # df['Gaussian_s7'] = gaussian_filter_s7
    # df['Variance_s3'] = variance_filter_s3   
    
 
    # #gabor filter
    # theta = [1,2]
    # sigma = [1,3]
    # lamda = list(np.arange(0, np.pi, np.pi / 4))
    # gamma = [0.05, 0.5]
    # gabor_vars = list(itertools.product(*[sigma,theta,lamda,gamma]))
    # gabor_kernel = list(map(createGaborKernel, gabor_vars))
    # num = 1
    # for kernel in gabor_kernel:
    #     Gabor = "Gabor_" + str(num)
    #     df[Gabor] = cv2.filter2D(np.float32(pixel_values_reshaped), cv2.CV_8UC3, kernel).reshape(-1)
    #     num += 1
            
    # create masks
    # mask_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\Masks\Masks_V2'  
    # os.chdir(mask_path) 
    # masks = os.listdir(mask_path)

    # df2 = pd.DataFrame()
    # label_value = list(map(imageDataframe, masks))
    # label_value = list(itertools.chain(*label_value))
    # df2['Label_Value'] = label_value
    
    # print("Completed D.A.S")
 
    # #########################################################
    # # DATA FORMATTING SECTION
    
    # print("Began D.F.S")
    
    # dataset = pd.concat([df, df2], axis =1)
 
   
    # X = dataset.drop(labels = ["Label_Value"], axis =1)
    # #X = X.dropna() # only if less features than labels
   
    # Y = dataset["Label_Value"].values
    # #Y = Y.dropna() # only if more features than labels
    
    # print("Completed D.F.S")
    
    # #########################################################
    # # MACHINE LEARNING SECTION
    
    # print("Began M.L.S")
    
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size = 0.2, random_state = 20)
    
    # from sklearn.ensemble import RandomForestClassifier
    
    # model = RandomForestClassifier(n_estimators = 100, random_state=(42))
    
    # model.fit(X_train, y_train)
    
    # from sklearn import metrics
    # prediction_test = model.predict(X_test)
    
    # print ("Model Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
    
    # feature_imp = pd.Series(model.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
    
    # # print(X.columns)
    # # print(model.feature_importances_)
    
    # # fig = plt.figure(figsize = (9,6))
    # # ax1 = plt.susbplot(111)
    # # ax1.barh(model.feature_importances_,X.columns)
    # # plt.show()
    # # plt.pyplot.barh(range(X.columns), model.feature_importances_, align='center')
    # # plt.yticks(np.arange(X.columns))
    # # plt.xlabel("Feature importance")
    # # plt.ylabel("Feature")
    
    # print(feature_imp)
    
    # import pickle
    
    # os.chdir(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Segmentation Model')
    # #bone_segmentation_1: R1 with default image windowing  (images/masks)
    # #bone_segmentation_2: R1 with default image windowing  (images/masks) and R2 with wide windowing (images/masks)
    # #bone_segmentation_3: R1 with one window (R1 imag/R1 masks)
    # #bone_segmentation_4: R1 with one window (R1_WW300_WC600/R1 masks_V1), TS = 20, With GABOR & VARIANCE
    # #bone_segmentation_5: R1 with one window (R1_WW300_WC600 & R1 masks_V2), TS = 20, With GABOR & VARIANCE , Model Accuracy =  0.99956568452808
    # #bone_segmentation_6: R1 with one window (R1_WW40_WC400) & R1_masks_V2), TS = 20, With GABOR & VARIANCE, Model Accuracy =  0.9992984807707788
    # #bone_segmentation_7: R1 with one window (R1_WW40_WC80)  & R1_masks_V2), TS = 20, With GABOR & VARIANCE, Model Accuracy =  0.9938518637768298
    # #bone_segmentation_8: R1 with one window (R1_WW176_WC2400)  & R1_masks_V2), TS = 20, With GABOR & VARIANCE, Model Accuracy =  0.9995551853627637
    # #bone_segmentation_9: R1 with one window (R1_WW300_WC1500)  & R1_masks_V2), TS = 20, With GABOR & VARIANCE, Model Accuracy =  0.9994720669706758
    # #bone_segmentation_10: R1 with one window (R1_WW-400_WC1500)  & R1_masks_V2), TS = 20, With GABOR & VARIANCE, Model Accuracy =  0.9994402195025496
    
    # model_name = "bone_segmentation_model_10"
    # pickle.dump(model, open(model_name, 'wb'))
    
    # print("Completed M.L.S")