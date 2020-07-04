# Behavioral Cloning Project

# Project Description
--
In this project, udacity behavior cloning simulator was used to clone driving behavior and then trained a Convolutional Neural Network based on the data gathered by the simulator. It is a supervised regression problem between the car steering angles and the road images in front of a car.

Those images were taken from three different camera angles (from the center, the left and the right of the car).

The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to work in this problem domain.

As image processing is involved, the model is using Convolutional layers for automated feature engineering.

### Files included

- model.py The script used to create and train the model.
- drive.py The script to drive the car autonomously. 
- model.h5 The model weights.
- video.mp4 - Video output of car driving in autonomous mode
- Behavioral_Cloning.ipynb - Notebook file which I have used for data exploration and plotting.
- writeup_report.md - Writeup template
- images folder - It has images, png files that are referenced in writeup_report.md file

# Model Design
This section describes how the model was built.

## Data collection 
To collect human driving data, training mode was used on simulator. The car was driven around the track using a keyboard and mouse for about 3 laps and driving data was recorded. Then the direction was reversed and another 2 laps were recorded. In addition a few short recordings were made of tricky parts of the track. The data recorded in the reverse direction ensured that the model did not simply memorize the conditions of the forward lap and generalized well to test-time conditions. Some sharps turn were tricky and initial models were really when negotiating them. The additional recordings helped the model stay in the middle of the road.

The simulator recorded images taken from the perspective of 3 cameras mounted at the front of the car at the left, center and a right of the hood. Along with these images the simulator also records the driving parameters at the instant an image was captured. These included steering angle, throttle and speed and were written to **driving_log.csv**. The images below show the same potion of the track from the left, center and right camera's perspective, respectively.

* **left**

![left camera](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/left.jpg)
* **center**

![center camera](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/center.jpg)
* **right**

![right camera](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/right.jpg)


## Data Preprocessing:

- Images are cropped so that the model won’t be trained other than the driving lane
- the images are resized to 66x200 (3 YUV channels) as per NVIDIA model
- GaussianBlur - to Blur image, to smoothing image and remove noise
- the images are normalized

```
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
```

![Preprocessed Image](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/preprocessed.png)  

## Image Augmentation:

For training, I used the following augmentation technique along with Python generator to generate unlimited number of images:

** Library used imgaug **
`pip install imgaug`


- Zooming will allow our model for better feature extraction
    *  `zoom(image)`
        ![zoom image](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/zoomed.png)
		
- Panning - Image pan is vertical and horizontal translation of images
translate percentage set 10% left and right as well 10% up and down
    *  `pan(image)`
       ![Pan image](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/panned.png)
	   
- Randomly altering image brightness (lighter or darker)
   *  `img_random_brightness(image)`    
       ![modified brightness image](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/brightness.png)
	   
- Flip steering angle for left to positive and right to negative,
It would be useful to create balanced distribution of left and right angles
steering angle
    * `img_random_flip(image, steering_angle)`
       ![random flip](images/flipped.png)

- `random_augment(image,steering_angle)` It is found that combined augmentation with variety of data will help model better generalize
Code each augment will only be applied 50% on new image
    - Randomly choose right, left or center images.
    - Randomly choose zoomed image 50%
    - Randomly flip image left/right 50%
    - Randomly altered brightness image 50%
    - Randomly translate image vertically 50%

![augmented_image](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/augmented.png)

- Batch generator function will take input data, create defined number of augmented images with label
- Benefit is it generates augmented images on fly avoid bottleneck memory space issue.
`batch_generator(image_paths, steering_ang, batch_size, istraining)`

## Model Architecture Design

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA for the end-to-end self driving test.  As such, it is well suited for the project.  

It is a deep convolution network which works well with supervised image classification / regression problems.  As the NVIDIA model is well documented, I was able to focus on how to adjust the training images to produce the best result with some adjustments to the model to avoid over fitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model.

- I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle.  However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction.  Overall, the model is very functional to clone the given steering behavior.  

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |convolution2d_1  |
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_2  |
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_3  |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_4  |
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |convolution2d_5  |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |252219  |                 |


### Training and training

## Training, Validation and Test

Images were split into train and validation set in order to measure the performance at every epoch.  Testing was done using the simulator.

Training and validation set are balanced.

![Training_Validation](https://github.com/alam121/Vehcile-Detection-using-HOG/blob/master/image/trainLossVsValidationloss.png)
As for training,

- I used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.
- I used Adam optimizer for optimization with learning rate of 1e-3.
- epoch 10
- steps_per_epoch=300
- validation_steps=200
- batch size 200 for training
- batch size 100 for validation
- Keras `fit_generator` method is used to train images generated by the generator

### Model results:
'''

Epoch 1/10
300/300 [==============================] - 282s 941ms/step - loss: 0.1412 - val_loss: 0.0727

Epoch 2/10
300/300 [==============================] - 276s 920ms/step - loss: 0.0785 - val_loss: 0.0571

Epoch 3/10
300/300 [==============================] - 279s 928ms/step - loss: 0.0633 - val_loss: 0.0461

Epoch 4/10
300/300 [==============================] - 278s 926ms/step - loss: 0.0562 - val_loss: 0.0406

Epoch 5/10
300/300 [==============================] - 273s 910ms/step - loss: 0.0514 - val_loss: 0.0404

Epoch 6/10
300/300 [==============================] - 274s 914ms/step - loss: 0.0486 - val_loss: 0.0359

Epoch 7/10
300/300 [==============================] - 274s 915ms/step - loss: 0.0466 - val_loss: 0.0361

Epoch 8/10
300/300 [==============================] - 272s 907ms/step - loss: 0.0450 - val_loss: 0.0353

Epoch 9/10
300/300 [==============================] - 279s 930ms/step - loss: 0.0434 - val_loss: 0.0349

Epoch 10/10
300/300 [==============================] - 279s 930ms/step - loss: 0.0415 - val_loss: 0.0332

'''


**Val_loss 0.0332 which is less than training loss 0.0415**


Car able to drive in autonomous mode video is in workspace `video.mp4`



