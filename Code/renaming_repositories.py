# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 08:57:51 2021

@author: pilovan
"""

import os 

def renameFunction(old_name, new_name):
    try:
        os.rename(old_name, new_name)
    except FileExistsError:
        print("File already Exists")
        print("Removing existing file")
        # skip the below code
        # if you don't' want to forcefully rename
        os.remove(new_name)
        # rename it
        os.rename(old_name, new_name)
        print('Done renaming a file')        

def rename_repository_2():
    #renaming repository 2
    filepath = r'C:\Users\pilovan\Documents\Image Processing\respository2\Shoulder_PNG'
    os.chdir(filepath)
    files = os.listdir(filepath)
    for file in files:
        #work only on odds
        if (int(file[3:-4]) > 1200 and int(file[3:-4]) < 1400) and int(file[3:-4]) % 2 != 0:
            file_new = file[3:-4]
            file_new = "R2_" + file_new + ".png"
            old_name = filepath + "/" + file
            new_name = filepath + "/" + file_new
            print(old_name)
            print(new_name)
            
            rename_function(old_name, new_name)
            
    print ("Completed")

def rename_repository_1():
    # rename repository 1
    filepath = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Combined_1_2\PNG'
    os.chdir(filepath)
    files = os.listdir(filepath)
    
    for file in files:
        file_new = file.split('_')
        file_new = file_new[2][:3]
        file_new = "R1_" + file_new + ".png"
        old_name = filepath + "/" + file
        new_name = filepath + "/" + file_new 
        print(old_name)
        print(new_name)
        
        rename_function(old_name, new_name)    
            
    print ("Completed")

def rename_mask_repositories_combined():
    filepath = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Combined_1_2\Masks'
    os.chdir(filepath)
    files = os.listdir(filepath)
    print(files)
    
    for file in files:
        if file[:2] == "R2":
            print(file)
            # file_new = "R2_" + file
            # file_new = filepath + "/" + file_new
            # file_old = filepath + "/" + file
            # print(file_new)
            # print(file_old)
            # rename_function(file_old, file_new)
            
        # if len(file) == 28:
        #     file_new = "R1_" + file[11:]
        #     file_new = filepath + "/" + file_new
        #     file_old = filepath + "/" + file
        #     print(file_new)
        #     print(file_old)
        #     rename_function(file_old, file_new)

def renameR1Images():
    filepath = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository\Radiogenomics\CT_InsNum_Sorted'
    os.chdir(filepath)
    files = os.listdir(filepath)
    for file in files:
        file_new = file.split('_')
        file_new = file_new[2]
        file_new = "R1_" + file_new
        file_new = filepath + "/" + file_new
        file_old = filepath + "/" + file
        renameFunction(file_old, file_new)     
    
def main():
    renameR1Images()
    # rename_mask_repositories_combined()
    # rename_repository_1()
    # rename_repository_2()
    
    print("Completed")
    
    
main()          
