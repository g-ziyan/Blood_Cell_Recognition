# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:33:23 2018

@author: sun_y
"""

dirpath = 'E:\\term3\\eece-523\\blood-cell-recognition' # Path of the code
import os
import datetime
os.chdir(dirpath)
import pandas as pd
from helpers.helpers_train import create # initialization
from helpers.helpers_train import changeImageName ## preparing jobs
from helpers.helpers_train import ProcessImage ## part I
from helpers.helpers_train import get_data ## part II

#from helpers.helpers_train import readImage

##
# Directories for parts
##

### dir of preparing jobs
dirraw_EOSINOPHILcell = 'TRAIN//Before//EOSINOPHIL'
celltype_EOSINOPHIL = "EOSINOPHIL"
dirraw_LYMPHOCYTEcell = 'TRAIN//Before//LYMPHOCYTE'
celltype_LYMPHOCYTE = "LYMPHOCYTE"
dirraw_MONOCYTEcell = 'TRAIN//Before//MONOCYTE'
celltype_MONOCYTE = "MONOCYTE"
dirraw_NEUTROPHILcell = 'TRAIN//Before//NEUTROPHIL'
celltype_NEUTROPHIL = "NEUTROPHIL"
dirraw_EOSINOPHILcelltest = 'TEST//Before//EOSINOPHIL'
celltype_EOSINOPHILtest = "EOSINOPHIL"
dirraw_LYMPHOCYTEcelltest = 'TEST//Before//LYMPHOCYTE'
celltype_LYMPHOCYTEtest = "LYMPHOCYTE"
dirraw_MONOCYTEcelltest = 'TEST//Before//MONOCYTE'
celltype_MONOCYTEtest = "MONOCYTE"
dirraw_NEUTROPHILcelltest = 'TEST//Before//NEUTROPHIL'
celltype_NEUTROPHILtest = "NEUTROPHIL"

## dir of partI
dirTrain = 'TRAIN\\Before'
dirTest = 'TEST\\Before'

## dir of partII
dirTrain_after = 'TRAIN\\After'
dirTest_after = 'TEST\\After'

###################################
## preparing jobs-- rename pictures
###################################
#changeImageName(dirraw_EOSINOPHILcell,celltype_EOSINOPHIL)
#changeImageName(dirraw_LYMPHOCYTEcell,celltype_LYMPHOCYTE)
#changeImageName(dirraw_MONOCYTEcell,celltype_MONOCYTE)
#changeImageName(dirraw_NEUTROPHILcell,celltype_NEUTROPHIL)


#changeImageName(dirraw_EOSINOPHILcelltest,celltype_EOSINOPHILtest)
#changeImageName(dirraw_LYMPHOCYTEcelltest,celltype_LYMPHOCYTEtest)
#changeImageName(dirraw_MONOCYTEcelltest,celltype_MONOCYTEtest)
#changeImageName(dirraw_NEUTROPHILcelltest,celltype_NEUTROPHILtest)


################################
## part I processing data
################################
# ProcessImage(dirTrain,dirTrain_after)
#ProcessImage(dirTest,dirTest_after)

################################
## part II get_data
################################

starttime = datetime.datetime.now()

X_train, y_train, z_train = get_data(dirTrain_after,5,2)
X_test, y_test, z_test = get_data(dirTest_after,5,2)
endtimea = datetime.datetime.now()
a_time = (endtimea -starttime).seconds
print('time_needed:',(endtimea - starttime).seconds)


print("Train X Shape --> ",X_train.shape)
print("Train y Shape --> ",y_train.shape)
print("Train z Shape --> ",z_train.shape)