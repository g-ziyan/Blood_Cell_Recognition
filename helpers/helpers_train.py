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


    
##
# Checking and creating directory
##
def create(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

#########################
### partI read image
#########################

def readImage(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images