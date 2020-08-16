import csv
#import cv2
import numpy as np
from scipy import ndimage
#import matplotlib.pyplot as plt

# read in csv file
lines=[]
with open('CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
        

# line[0] has only strings
lines=lines[1:]
#lines=lines[1:1000]



      
samples=np.array(lines)        

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#split date in train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        # first shuffle dataset
        shuffle(samples) #random.shuffle(sample)
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                
                #source_path for center left and right image
                source_path_center = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
                
                filename_center = source_path_center.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]
                currentpath_center='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_center
                currentpath_left='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_left
                currentpath_right='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_right
                
                #loading images
                image_center = ndimage.imread(currentpath_center)
                image_left = ndimage.imread(currentpath_left)
                image_right = ndimage.imread(currentpath_right)
                
                # add a correction factor for left and right camera image
                measurement_center=float(batch_sample[3])
                measurement_left=float(batch_sample[3])+0.20
                measurement_right=float(batch_sample[3])-0.20
                
                #append all images and steering angles
                #images.append(image_center)
                images.extend([image_center,image_left,image_right])
                #angles.append(measurement_center)
                angles.extend([measurement_center,measurement_left,measurement_right])
                #images.append(image_left)
                #measurements.append(measurement_left)
                #images.append(image_right)
                #measurements.append(measurement_right)
                
                ##also append mirrored images
                images.append(np.fliplr(image_center))
                #append also inverted steering angle
                angles.append(-1.0*measurement_center)
                images.append(np.fliplr(image_left))    
                angles.append(-1.0*measurement_left)    
                images.append(np.fliplr(image_right))
                angles.append(-1.0*measurement_right)
 
            
            X_train = np.array(images)
            y_train = np.array(angles)
            #return batch data every time the generator is called
            yield sklearn.utils.shuffle(X_train, y_train)



'''
## Variant without generator

# read in images and steering angles
for line in lines:
    #source_path for center left and right image
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]

    filename_center = source_path_center.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    currentpath_center='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_center
    currentpath_left='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_left
    currentpath_right='CarND-Behavioral-Cloning-P3/data/IMG/' + filename_right


    #loading images
    image_center = ndimage.imread(currentpath_center)
    image_left = ndimage.imread(currentpath_left)
    image_right = ndimage.imread(currentpath_right)
    
    
    
    # add a correction factor for left and right camera image
    measurement_center=float(line[3])
    measurement_left=float(line[3])+0.1
    measurement_right=float(line[3])-0.1
    
    #append all images and steering angles
    images.append(image_center)
    measurements.append(measurement_center)
    images.append(image_left)
    measurements.append(measurement_left)
    images.append(image_right)
    measurements.append(measurement_right)    
    
    
    ##also append mirrored images
    images.append(np.fliplr(image_center))
    #append also inverted steering angle
    measurements.append(-1.0*measurement_center)
    
    images.append(np.fliplr(image_left))    
    measurements.append(-1.0*measurement_left)    
    images.append(np.fliplr(image_right))
    measurements.append(-1.0*measurement_right)
    
    
#make training date a numpy array
X_train=np.array(images)
y_train=np.array(measurements)
print(X_train.shape)
print(y_train.shape)

#print(y_train)
 
 '''     
 

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D    


##define Keras model
model = Sequential()
#nomalizing image data
model.add(Lambda(lambda x: x/255 -0.5, input_shape=(160, 320, 3)))
#cutting of some lines at bottom and top of the image
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

## fit model to data
model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7, verbose=1)

from math import ceil

batch_size=64 #every dataset within the batch consists of 6 iamges
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

#save trained model
model.save('model2.h5')
print("----------------------------------model saved-----------------------------")

## plotting loss over epochs
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()




