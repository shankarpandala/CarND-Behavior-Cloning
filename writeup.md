# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

* #### Note: Most of the code is taken from the classroom lessons and is modified to fit my requirements. Architecture is taken from Nvidia paper which explained in next sections

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

* I have used an architecture that is used in [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) paper by Nvidia
* I have used dropout layers of 50% probability to handle over fitting.
* In addition to the architecture from the paper, I have added a Cropping2D layer to crop the images to contain only road

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers of 50% probability to handle overfitting at every convolution layer and dense layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used the data provided by udaciy as simulator is not smooth in the workspace. 

Here is the sample training data that show how the left, center and right cameras are capturing the images of the road

| Left Camera Image  | Center Camera Image | Right Camera Image |
| ------------- | ------------- | ------------- |
| ![Center Image](data/IMG/left_2016_12_01_13_30_48_287.jpg) | ![Center Image](data/IMG/center_2016_12_01_13_30_48_287.jpg)  | ![Center Image](data/IMG/right_2016_12_01_13_30_48_287.jpg)  |


There are a total of 8036 data points out of which 80% is used for training and 20% is used for validaation

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropout layers.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I increased the epochs to train more.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py) is a modified model from Nvidia paper as mention above

Here is the architecture 

| Layer         | Description                                                 |
|---------------|-------------------------------------------------------------|
| Normalization | lambda x: x / 255 - 0.5                                     |
| Cropping      | Cropping2D(cropping=((70, 25), (0, 0))                      |
| Convolution   | 24, 5x5 kernels, 2x2 stride, valid padding, RELU Activation |
| Dropout       | regularization 0.5                                          |
| Convolution   | 36, 5x5 kernels, 2x2 stride, valid padding, RELU Activation |
| Dropout       | regularization 0.5                                          |
| Convolution   | 48, 5x5 kernels, 1x1 stride, valid padding, RELU Activation |
| Dropout       | regularization 0.5                                          |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding, RELU Activation |
| Dropout       | regularization 0.5                                          |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding, RELU Activation |
| Dropout       | regularization 0.5                                          |
| Flatten       | Flatten Layer                                               |
| Dense         | Fully Connected Layer 100                                   |
| Dropout       | regularization 0.5                                          |
| Dense         | Fully connected Layer 50                                    |
| Dropout       | regularization 0.5                                          |
| Dense         | Fully connected Layer 10                                    |
| Dense         | Output Layer 1                                              |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. However, the simulator was lagging on workspace and makes me a bad driver. 

I have downloaded the simulator to the local machine and generated the data, size was huge. So, I decided to use the data provided by udacity.

Data Augumentation is done to flip the center images to reeduce the bias in the environment also a correction factor is induced to left and right images by +/- 0.2 .

I then preprocessed this data by normalizing the data by using a lambda layer lambda x: x / 255 - 0.5  


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by run1 on the simulator in autonomous mode without going off the track I used an adam optimizer so that manually training the learning rate wasn't necessary.
