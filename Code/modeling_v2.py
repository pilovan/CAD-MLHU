# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:20:10 2021

@author: pilovan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:25:07 2021

@author: pilovan

ML & DICOM
"""

## Import libraries
import sys
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

#windowing or contrast will matter
image = cv2.imread(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_PNG\000130.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_reshaped = image.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = image_reshaped

#plt.imshow(image, cmap=plt.cm.gray)

#generate gabor features 
num = 1 
#ksize = 9 # not in for loop be can be 
kernels = []
ksize = 30
for theta in range(2): 
    theta = theta / 4  * np.pi
    for sigma in (1,3):
        for lamda in np.arange(0, np.pi, np.pi / 4):
            for gamma in (0.05, .5):
                
                gabor_label = 'Gabor ' + str(num)
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma )
                kernels.append(kernel)
                
                filtered_image = cv2.filter2D(image_reshaped, cv2.CV_8UC3, kernel)
                filtered_image = filtered_image.reshape(-1)
                df[gabor_label] = filtered_image
                #print(gabor_label, ': theta= ', theta, ': sigma=', sigma, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  
                
from skimage.filters import roberts, sobel, scharr, prewitt                

edges = cv2.Canny(image, 100, 200)
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1

edges_roberts = roberts(image)
edges_roberts1 = edges_roberts.reshape(-1)
df['Roberts Edge'] = edges_roberts1

edges_sobel = sobel(image)
edges_sobel1 = edges_sobel.reshape(-1)
df['Sobel Edge'] = edges_sobel1

edges_scharr = scharr(image)
edges_scharr1 = edges_scharr.reshape(-1)
df['Scharr Edge'] = edges_scharr1

edges_prewitt = prewitt(image)
edges_prewitt1 = edges_prewitt.reshape(-1)
df['Prewitt Edge'] = edges_scharr1

from scipy import ndimage as nd

gaussian_image = nd.gaussian_filter(image, sigma = 3)
gaussian_image1 = gaussian_image.reshape(-1)
df['Gaussian S3'] = gaussian_image1

gaussian_image2 = nd.gaussian_filter(image, sigma = 7)
gaussian_image3 = gaussian_image.reshape(-1)
df['Gaussian S7'] = gaussian_image3

median_image = nd.median_filter(image, size = 3)
median_image1 = median_image.reshape(-1)
df['Median S3'] = median_image1

# load in labeled image
#image_labeled = cv2.imread(r'C:\Users\pilovan\Documents\Image Processing\Body_masked.tif')
image_labeled = cv2.imread(r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Results\Masks\Bone_mask.png')


image_labeled = cv2.cvtColor(image_labeled, cv2.COLOR_BGR2GRAY)
image_labeled1 = image_labeled.reshape(-1)
df['Label_Value'] = image_labeled1
#df = df[df.Label_Value != 0]
plt.imshow(image_labeled)

Y = df['Label_Value'].values
X = df.drop(labels = ['Label_Value'], axis = 1)

# # # modeling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .4 , random_state = 42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

model.fit(X_train, y_train)

prediction_test = model.predict(X_test)

from sklearn import metrics
print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index = features_list).sort_values(ascending = False)

print(feature_imp)


# saves model through pickle library 
import pickle

# named kidney model 
# saving model via writing in binary
pickle.dump(model, open('HU_segmentation_model', 'wb'))

# loads model via reading in binary 
load_model = pickle.load(open('HU_segmentation_model', 'rb'))
result = load_model.predict(X)

# shape result as original image 
segmented = result.reshape(image.shape)

rows = 2
columns = 2
fig = plt.figure(figsize=(50,50))

fig.add_subplot(rows, columns, 1)
plt.imshow(image, cmap=plt.cm.gray) 
plt.title("Test",fontsize=40)

fig.add_subplot(rows, columns, 2)
plt.imshow(segmented, cmap=plt.cm.gray) 
plt.title("Result",fontsize=40)


# #overlay ml result with orginal image 
# #https://theailearner.com/2019/03/26/image-overlays-using-bitwise-operations-opencv-python/
# image = cv2.merge([image,image,image])

# segmented_layered = cv2.merge([segmented,segmented, segmented])
# segmented_layered[segmented_layered[:,:, 1] > 0] = [32,44,255]



# rows, cols, channels = segmented_layered.shape
# roi = image[0:rows, 0:cols]

# ret, mask = cv2.threshold(segmented, 10 ,255, cv2.THRESH_BINARY)
# mask_inverse = cv2.bitwise_not(mask)

# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inverse)

# img2_fg = cv2.bitwise_and(segmented_layered, segmented_layered,mask = mask )

# out_img = cv2.add(img1_bg, img2_fg)
# image[0:rows, 0:cols] = out_img

# cv2.imshow('image', image)
# cv2.imwrite('modeling_result.jpg', image)


# cv2.waitKey(0)
# cv2.destroyAllWindows