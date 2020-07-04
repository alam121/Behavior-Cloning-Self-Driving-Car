# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 12:27:24 2019

@author: samrat.pyaraka
"""
#downloading data from git
!git clone https://github.com/samratpyaraka/Data.git

#install imgaug package for data augmentation
!pip3 install imgaug

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random


class GenerateModelNDoDataAug():

    def __init__(self, pathOfDir):
        self.pathOfDir = pathOfDir
    
    
    """
        Load A CSV File
    """
    def loadCSVToDataFrame(self, csvFileName, doesFileHasColNames=True):
        
        if doesFileHasColNames:
            df = pd.read_csv(os.path.join(self.pathOfDir, csvFileName))
        else:
            columns = ['center', 'left', 'right', 'steering', 'throttle', 
                   'reverse', 'speed']
            df = pd.read_csv(os.path.join(self.pathOfDir, csvFileName),
                         names = columns)
            
        pd.set_option('display.max_colwidth', -1)
        
        for index, row in df.iterrows():
    
            df.at[index,'center'] = self.process_fileNames(row['center'])
            df.at[index,'left'] = self.process_fileNames(row['left'])
            df.at[index,'right'] = self.process_fileNames(row['right'])
        
        return df
    
    """
        Remove Directory structure and only store file name in csv
    """    
    def process_fileNames(self,name):
        
        name = name.split('\\')
        name = name[-1:]
        processedFileName  = '/'.join(name) 
        return processedFileName
    
    """
        Data is collected from simulator has both left steering and right steering data 
        Data has mostly 0 steering angle we need to remove that data
    """
    def balanceData(self, df):
        
        num_bins = 25
        
        # Limit I choose above which all other data will be deleted
        samples_per_bin = 400 
        hist, bins = np.histogram(df['steering'], num_bins)
        center = (bins[:-1]+ bins[1:]) * 0.5
        
        remove_list = []
        for j in range(num_bins):
          list_ = []
          for i in range(len(df['steering'])):
            if df['steering'][i] >= bins[j] and df['steering'][i] <= bins[j+1]:
              list_.append(i)
          list_ = shuffle(list_)
          list_ = list_[samples_per_bin:]
          remove_list.extend(list_)
        
        df.drop(df.index[remove_list], inplace=True)
        
        return df
    
    """
        Function to load image data along with steering angle
        Left image steering angle adjusted by 0.15
        Right image steering angle adjusted by -0.15
    """
    def load_img_steering(self, pathToFile, df):
        image_path = []
        steering = []
        pathToImgFile = self.pathOfDir + pathToFile
      
        for i in range(len(df)):
            indexed_data = df.iloc[i]
            center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
            image_path.append(os.path.join(pathToImgFile, center.strip()))
            steering.append(float(indexed_data[3]))
            # left image append
            image_path.append(os.path.join(pathToImgFile,left.strip()))
            steering.append(float(indexed_data[3])+0.15)
            # right image append
            image_path.append(os.path.join(pathToImgFile,right.strip()))
            steering.append(float(indexed_data[3])-0.15)
        
        image_paths = np.asarray(image_path)
        steerings = np.asarray(steering)
        return image_paths, steerings 
  
    """
        Data augmentation is technique of creating new variety of dataset with existing dataset
        I have used imgaug library for data augmentation instead of Keras which add more flexibility
        please make sure to install pip install imgaug 
        Zooming will allow our model for better feature extraction  
        scale 1,1.3 states we are zooming in and we can zoom 30% of entire image
    """
    def zoom(self, image):
        zoom = iaa.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
        return image
  
    
    """
        Image pan is vertical and horizontal translation of images 
        translate percentage set 10% left and right as well 10% up and down
    """
    def pan(self, image):
        pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        return image
    
    
    """
        Function randomly alter brightness of image 
        We are interested in darker image it does by multiplying pixel intensity 
        from below range of values
    """
    def img_random_brightness(self, image):
        brightness = iaa.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
        return image
    
    
    """
        Function to flip steering angles for left to positive and for right to negative.
        It would be useful to create balanced distribution of left and right angle steering data.
    """
    def img_random_flip(self, image, steering_angle):
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
        return image, steering_angle
    
    
    """
        Below function used to randomize augmentation 
        It is found that combined augmentation with variety of data will help model better generalize
        Code each augment will only be applied 50% on new image.
    """
    def random_augment(self, image, steering_angle):
        image = mpimg.imread(image)
        if np.random.rand() < 0.5:
          image = self.pan(image)
        if np.random.rand() < 0.5:
          image = self.zoom(image)
        if np.random.rand() < 0.5:
          image = self.img_random_brightness(image)
        if np.random.rand() < 0.5:
          image, steering_angle = self.img_random_flip(image, steering_angle)
        
        return image, steering_angle
        
    """
        - Preprocessing Techniques
        - Cropping image - Unnecessary feature will be removed
        - Convert to YUV - As per Nvidia Paper convert RGB to YUV 
        - GaussianBlur - to Blur image to smoothing image and remove noise
        - resize 200,66 - As we choose build Nvidia model which take size 200,66
        - Normalization - Divide it by 255
    """
    def img_preprocess(self, img):
        img = img[60:135,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img/255
        return img
    
    """
        Batch generator function will take input data , create defined number 
        of augmented images with label.
        Benefit It generate augmented images on fly avoid bottleneck memory space issue.
    """
    def batch_generator(self, image_paths, steering_ang, batch_size, istraining):
      
      while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
          random_index = random.randint(0, len(image_paths) - 1)
          
          if istraining:
            im, steering = self.random_augment(image_paths[random_index], steering_ang[random_index])
         
          else:
            im = mpimg.imread(image_paths[random_index])
            steering = steering_ang[random_index]
          
          im = self.img_preprocess(im)
          batch_img.append(im)
          batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))
        
    """
        Function to create a model provided by NVIDIA.
    """    
    def convModelByNVIDIA(self):
        model = Sequential()
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
        model.add(Convolution2D(64, 3, 3, activation='elu'))
      
        model.add(Convolution2D(64, 3, 3, activation='elu'))
    
        model.add(Flatten())
      
        model.add(Dense(100, activation = 'elu'))
      
        model.add(Dense(50, activation = 'elu'))
      
        model.add(Dense(10, activation = 'elu'))
     
        model.add(Dense(1))
      
        optimizer = Adam(lr=1e-3)
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    
dataDir = 'Data'
fileName = 'driving_log2.csv'
# Create object of the GenerateModelNDoDataAug Class
ModelObject = GenerateModelNDoDataAug(dataDir)

#Get the DataFrame of the CSV where all the image path and driving logs are stored
df = ModelObject.loadCSVToDataFrame(fileName)

#balance the data as most of the value of steering are 0
balancedData = ModelObject.balanceData(df)

#get image path and steering values
image_paths, steerings = ModelObject.load_img_steering('/IMG', df)

# split the data in training and validation in a ratio of 80% and 20%
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# get the model
Model = ModelObject.convModelByNVIDIA()
print(Model.summary())

#train the model 
history = Model.fit_generator(ModelObject.batch_generator(X_train, y_train, 200, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=ModelObject.batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)
								  
#plotting histogram of training loss vs validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

#save the model
Model.save('model.h5')
    
