import os
import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0) #Removing row names of csv file
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    
                    if i == 0 :
                        images.append(cv2.flip(image,1)) #flip center image
                        angles.append(angle*-1.0)
                    elif i == 1: #left image steering angle correction
                        images.append(image)
                        angles.append(angle + 0.2)
                    elif i == 2: #right image steering angle correction
                        images.append(image)
                        angles.append(angle - 0.2)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D, Dropout, Flatten, Dense, Lambda, MaxPooling2D
from sklearn.utils import shuffle
model = Sequential()

model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)))
model.add(Cropping2D(((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 
                    len(train_samples), validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)
model.save('model.h5')
