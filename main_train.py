# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:33:23 2018

@author: sun_y
"""

dirpath = 'E:\\term3\\eece 523\\blood cell recognition' # Path of the code
import os
import datetime
os.chdir(dirpath)
import pandas as pd
from helpers.helpers_train import create # initialization
from helpers.helpers_train import readImage ## part I

##
# Directories for parts
##
dirrawcell = 'output/train_data/'

######################
## part I read image
######################
starttime = datetime.datetime.now()

#if not os.path.exists(dirrawcell):
    
images = readImage(dirrawcell)
endtimea = datetime.datetime.now()
a_time = (endtimea -starttime).seconds
print('time_needed:',(endtimea - starttime).seconds)