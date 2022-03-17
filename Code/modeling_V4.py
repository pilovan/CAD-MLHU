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
from matplotlib import pyplot as plt
import pickle

        
if __name__ == '__main__':

    #########################################################
    # DATA FORMATTING SECTION    
    segmentation_dir = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Bone_Classification_Data'
    os.chdir(segmentation_dir)
    filename = "Bone_Classification_Dataset.pkl"
    
    print("Importing Data")
    dataset = pd.read_pickle(filename)

    X = dataset.drop(labels = ["Label_Value"], axis =1)
    Y = dataset["Label_Value"].values
    Y = np.where(Y > 0, 1, Y)
    
    #########################################################
    # MACHINE LEARNING SECTION
    
    print("Began M.L.S")
    
    from sklearn import metrics    
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.model_selection import train_test_split
    
    print("Splitting Data 80/20")
    X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size = 0.2, random_state = 20)
    
    # n_estimators = [1,2,4,8,16,32,64,100,200]
    # n_depths = range(1,50,5)
    train_results = []
    test_results = []
    
    model_name = "RFC_Bone_Estimator_100_Depth_36_withOOB"
    filepath = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\RFC_Models\N_depths' + '/' + model_name
   
    # files = os.listdir(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\RFC_Models\N_depths')
    # print("The current files in the directory: ", files)
    
    # if model_name in files:
    #     model = pickle.load(open(filepath, 'rb'))
    #     print("Opened Previously Created Model: " + model_name)
    # else:
    #     print("Did not open model")
    
    model = RandomForestClassifier(n_estimators=100, max_depth = 36, oob_score=True, n_jobs = -1, random_state=(42))

    print("Fitting Dataset")
    model.fit(X_train, y_train)
    
    train_prediction = model.predict(X_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_prediction)
    roc_auc = metrics.auc(fpr,tpr)
    train_results.append(roc_auc)
    
    y_prediction = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prediction)
    roc_auc = metrics.auc(fpr,tpr)
    test_results.append(roc_auc)
    
    print ("Model Accuracy = ", metrics.accuracy_score(y_test, y_prediction))
    feature_imp = pd.Series(model.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
    print(feature_imp)

    from matplotlib.legend_handler import HandlerLine2D
    
    line1, = plt.plot(1, train_results, 'b', label= "Train AUC")
    line2, = plt.plot(1, test_results, 'r', label= "Test AUC")
    
    plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC Score')
    plt.xlabel('Tree Depth')
    plt.show()
    
    pickle.dump(model, open(filepath, 'wb'))
    
    #n_estimators no max depth
    