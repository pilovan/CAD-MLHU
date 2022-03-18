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

def renameFiles(filepath, repository):
    prefix = "R1_"
    suffix = ".png"

    files = os.listdir(filepath)
    for file in files:
        new_name = file.split('_')
        new_name = new_name[1][:3]
        new_name = prefix + new_name + suffix
        new_name = filepath + "/" + new_name 
        old_name = filepath + "/" + file
        
        print("Old Name: {} \n".format(old_name))
        print("New Name: {} \n".format(new_name))
        
        renameFunction(old_name, new_name)    
    
def main():
    filepath = r'C:\Users\pilovan\Documents\GitHub\DICOM-ML\Repository'
    renameFiles(filepath)
    
    print("Renaming Completed")
    
    
main()          
