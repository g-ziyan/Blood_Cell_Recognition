# -*- coding: utf-8 -*-

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
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D,MaxPooling2D,Flatten


import sklearn

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
### partI process image
#########################

def ProcessImage(folder, folder_after):
    
    for cellname in tqdm(os.listdir(folder)):
        #print(cellname)
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
                img_file = cv2.imread(os.path.join(folder,wbc_type,image_filename)) ## read as a gray image
                image = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
                denoised = filters.median(image)
                if img_file is not None:
                    img_file_resize = scipy.misc.imresize(arr=denoised, size=(60, 80))
                    img_arr = np.asarray(img_file_resize)
                    X.append(img_arr)
                    y.append(label)
                    z.append(label2)
    
    
    X = np.asarray(X)/255.0
    y = np.asarray(y)
    z = np.asarray(z)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    
    y_Hot = to_categorical(y, y_num_classes)
    z_Hot = to_categorical(z, z_num_classes)
    
    return (X,y_Hot,z_Hot)



################################
### part III train data
################################
def runKerasCNNAugment(a,b,c,d,e):
   # batch_size = 128
    num_classes = len(b[0])
    epochs = 30
#     img_rows, img_cols = a.shape[1],a.shape[2]
    img_rows,img_cols=60,80
    input_shape = (img_rows, img_cols,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape = input_shape,
                     strides=e))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    
    history = model.fit_generator(datagen.flow(a,b, batch_size = 32),
                        steps_per_epoch=len(a) / 32, epochs=epochs)

    score = model.evaluate(c,d, verbose=0)
    
    return(score)


dirTrain_after = '../input/dataset2-master/dataset2-master/images/TRAIN/'
dirTest_after = '../input/dataset2-master/dataset2-master/images/TEST/'

X_train, y_train, z_train = get_data(dirTrain_after,5,2)
X_test, y_test, z_test = get_data(dirTest_after,5,2)

x_train = X_train.reshape(len(X_train), 60, 80, 1)
x_test = X_test.reshape(len(X_test), 60, 80, 1)
runKerasCNNAugment(x_train,y_train,x_test,y_test,1)
