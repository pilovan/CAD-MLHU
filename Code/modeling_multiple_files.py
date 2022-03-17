# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:32:00 2021

@author: pilovan
"""
 
import numpy as np
import cv2
import pandas as pd
import os

# image_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted_PNG'
image_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted_PNG'
os.chdir(image_path)

image_dataset = pd.DataFrame()
image_name = []
count = 0
files = os.listdir(image_path)
for image in files[:2]:
    # if count % 2 == 0:
    image_name = image.split('_')
    # if int(image_name[1]) % 2 != 0:
    # if count < 5:

        # read images 
    print(image)
    
    df = pd.DataFrame()  
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #skip RGB checking, we know what it is 
        
    # add data to df 
    
    pixel_values = img.reshape(-1)
    df['Pixel_Values'] = pixel_values

    df['Image_Name'] = image


    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    #print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)  
                    print(kernel)
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
    # ########################################
    # #Gerate OTHER FEATURES and add them to the data frame
                    
    # #CANNY EDGE
    # edges = cv2.Canny(img, 100,200)   #Image, min and max values
    # edges1 = edges.reshape(-1)
    # df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    # #SOBEL
    # edge_sobel = sobel(img)
    # edge_sobel1 = edge_sobel.reshape(-1)
    # df['Sobel'] = edge_sobel1
    
    # #SCHARR
    # edge_scharr = scharr(img)
    # edge_scharr1 = edge_scharr.reshape(-1)
    # df['Scharr'] = edge_scharr1
    
    # #PREWITT
    # edge_prewitt = prewitt(img)
    # edge_prewitt1 = edge_prewitt.reshape(-1)
    # df['Prewitt'] = edge_prewitt1
    
    # #GAUSSIAN with sigma=3
    # from scipy import ndimage as nd
    # gaussian_img = nd.gaussian_filter(img, sigma=3)
    # gaussian_img1 = gaussian_img.reshape(-1)
    # df['Gaussian s3'] = gaussian_img1
    
    # #GAUSSIAN with sigma=7
    # gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    # gaussian_img3 = gaussian_img2.reshape(-1)
    # df['Gaussian s7'] = gaussian_img3
    
    # #MEDIAN with sigma=3
    # median_img = nd.median_filter(img, size=3)
    # median_img1 = median_img.reshape(-1)
    # df['Median s3'] = median_img1
    
    # #VARIANCE with size=3
    # variance_img = nd.generic_filter(img, np.var, size=3)
    # variance_img1 = variance_img.reshape(-1)
    # df['Variance s3'] = variance_img1  #Add column to original dataframe
   
    image_dataset = image_dataset.append(df)
    # count += 1
        
# ######################################
# # create same type of df for masks
# mask_dataset = pd.DataFrame()

# mask_path = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Combined_1_2\Masks'  
# os.chdir(mask_path)    
# mask_name = []
# count = 0
# masks = os.listdir(mask_path)
# for mask in masks[:20]:
#     # mask_name = mask.split("_")
#     # if int(mask_name[2]) % 2 == 0:
#     # if count % 2 == 0:
#     # if count < 2:
#     print(mask)
#     df2 = pd.DataFrame()
#     msk = cv2.imread(mask)
#     msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    
    
#     label_values = msk.reshape(-1)
#     df2['Label_Value'] = label_values
#     df2['Mask_Name'] = mask
    
#     mask_dataset = mask_dataset.append(df2)
#     # count += 1
    

# print("finished importing images & masks")
# # image_dataset = image_dataset[~image_dataset.index.duplicated(keep='first')]
# # mask_dataset = mask_dataset[~mask_dataset.index.duplicated(keep='first')]

# # image_dataset = image_dataset.reset_index()
# # mask_dataset = mask_dataset.reset_index()
# dataset = pd.concat([image_dataset, mask_dataset], axis =1)



# #dataset = dataset[dataset.Label_Value != 0]

# X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis =1)

# Y = dataset["Label_Value"].values

# print("almost done")

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size = 0.4, random_state = 20)


# # os.chdir(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Troubleshooting')
# # y_train_excel = pd.DataFrame(y_train)
# # y_train_excel[0:50000].to_excel("MMF_V1.xlsx")  

# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators = 100, random_state=(42))

# model.fit(X_train, y_train)
# print("done fitting")

# from sklearn import metrics 
# prediction_test = model.predict(X_test)

# print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

# import pickle

# os.chdir(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Segmentation Model')
# #bone_segmentation_1: R1 with default image windowing  (images/masks)
# #bone_segmentation_2: R1 with default image windowing  (images/masks) and R2 with wide windowing (images/masks)
# #bone_segmentation_3: R1 with various windowing (images/masks) and R2 (masks only)
# model_name = "bone_segmentation_model_3_1"
# pickle.dump(model, open(model_name, 'wb'))

















# segmented = result.reshape((img.shape))

# from matplotlib import pyplot as plt
# rows = 2
# columns = 2
# fig = plt.figure(figsize=(50,50))

# fig.add_subplot(rows, columns, 1)
# plt.imshow(img, cmap=plt.cm.gray) 
# plt.title("Test DCM225",fontsize=40)

# fig.add_subplot(rows, columns, 2)
# plt.imshow(segmented, cmap=plt.cm.gray) 
# plt.title("Result DCM225 with DCM230 Training",fontsize=40)

# fig.add_subplot(rows, columns, 3)
# dcm126 = plt.imread(r"C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted_PNG\DCM_InsNum_230.png")
# plt.imshow(dcm126, cmap=plt.cm.gray) 
# plt.title("DCM230",fontsize=40)

# fig.add_subplot(rows, columns, 4)
# dcm126_bone_mask = plt.imread(r"C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Masks\DCM_InsNum_230_bone_mask.png")
# plt.imshow(dcm126_bone_mask, cmap=plt.cm.gray) 
# plt.title("DCM230 Bone Mask",fontsize=40)
# #plt.imsave('segmented_rock_RF_100_estim.jpg', segmented, cmap ='jet')

# img = cv2.merge([img,img,img])

# segmented_layered = cv2.merge([segmented,segmented, segmented])
# segmented_layered[segmented_layered[:,:, 1] > 0] = [32,44,255]

# rows, cols, channels = segmented_layered.shape
# roi = img[0:rows, 0:cols]

# ret, mask = cv2.threshold(segmented, 10 ,255, cv2.THRESH_BINARY)
# mask_inverse = cv2.bitwise_not(mask)

# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inverse)

# img2_fg = cv2.bitwise_and(segmented_layered, segmented_layered,mask = mask )

# out_img = cv2.add(img1_bg, img2_fg)
# img[0:rows, 0:cols] = out_img

# cv2.imshow('image', img)
# #cv2.imwrite('modeling_result.jpg', img)


# cv2.waitKey(0)
# cv2.destroyAllWindows
