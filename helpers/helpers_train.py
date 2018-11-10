# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:33:27 2018

@author: sun_y
"""

import os
import shutil # for concatenation of dataframes
import pandas as pd
import pickle # for saving Python objects
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import scipy
from scipy import ndimage
from keras.utils.np_utils import to_categorical
from PIL import Image,ImageEnhance,ImageFilter
from skimage import filters

#################
# Concatenation #
#################
##
# Concatenate 'folder/file1.txt', 'folder/file2.txt', etc. into 'folder.txt'.
##
def concatenate(folder):
    (_, _, filenames) = next(os.walk(folder))
    filenames = [folder + '/' + filename for filename in filenames]
    #filenames = filenames[:1]
    
    with open(folder + '.txt','wb') as wfd:
        for f in filenames:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd, 1024*1024*10)
                # 10MB per writing chunk to avoid reading big file into memory.


    
####################################
# Checking and creating directory
####################################
def create(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
######################################
### preparing jobs-- rename pictures
#####################################
def changeImageName(folder,celltype):
    images_path = folder
    image_list = os.listdir(images_path)
    for i,  image in enumerate(image_list):
#        ext = os.path.splitext(image)[1]
#        if ext == '.jpeg':
        src = images_path + '//' + image
        dst = images_path + '//' + celltype + str(i) + '.jpg'
        os.rename(src, dst)

#########################
### partI processing image
#########################

def ProcessImage(folder, folder_after):
    
    for cellname in os.listdir(folder):
        print(cellname)
        oldpath = os.path.join(folder,cellname)
        newpath = os.path.join(folder_after,cellname)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for filename in os.listdir(oldpath):
            image_original = cv2.imread(os.path.join(oldpath,filename))
            image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
            denoised = filters.median(image)
            imagenamenow = newpath + '//' + '_new' +  filename 
            cv2.imwrite(imagenamenow,denoised)
            




#########################
### partII get data
#########################

def get_data(folder, y_num_classes, z_num_classes):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    z = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 1
                label2 = 1
            elif wbc_type in ['EOSINOPHIL']:
                label = 2
                label2 = 1
            elif wbc_type in ['MONOCYTE']:
                label = 3  
                label2 = 0
            elif wbc_type in ['LYMPHOCYTE']:
                label = 4 
                label2 = 0
            else:
                label = 5
                label2 = 0
            for image_filename in tqdm(os.listdir(os.path.join(folder,wbc_type))):
                img_file = cv2.imread(os.path.join(folder,wbc_type,image_filename),0) ## read as a gray image
                if img_file is not None:
                    img_file_resize = scipy.misc.imresize(arr=img_file, size=(60, 80))
                    img_arr = np.asarray(img_file_resize)
                    X.append(img_arr)
                    y.append(label)
                    z.append(label2)
    
    
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    
    y_Hot = to_categorical(y, y_num_classes)
    z_Hot = to_categorical(z, z_num_classes)
    
    return (X,y_Hot,z_Hot)

