# **Behavioral Cloning** 

## Update 

### Changes since first submission

* implemented generator porperly
* adjusted correction factor for left and right camera to 0.2 (0.1 before)
* dropout layers only before every fully connected layer
* reduced epochs form 7 to 5
* adjusted writeup below


## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture and Design Approach

After starting with a very simple network of just one fully connected layer I tried out the Nvidia Network as shown in the lessen and kept doing little adaptations from there.

First I added a Lambda layer for normalizing and rescaling the data. then I added a cropping layer to cut off unimportant parts of the images. To prevent overfitting I added Dropout layers.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Lambda             	| x: x/255 -0.5 	                            |
| Cropping2D			| cropping top 60 and botom 20 lines			|
| Conv2D    	      	| 5x5 filter, 2x2 stride, depth 24, activation relu     		|
| Conv2D    	      	| 5x5 filter, 2x2 stride, depth 36, activation relu     		|
| Conv2D    	      	| 5x5 filter, 2x2 stride, depth 48, activation relu     		|
| Conv2D    	      	| 3x3 filter, depth 64, activation relu    		|
| Conv2D    	      	| 3x3 filter, depth 64, activation relu    		|
| Dropout       	    | dropout probability 0.5                    	|
| Flatten          	    |       								    	|
| Fully connected      	| output 100                		     		|
| Dropout       	    | dropout probability 0.5                    	|
| Fully connected      	| output 50                 		     		|
| Dropout       	    | dropout probability 0.5                    	|
| Fully connected      	| output 10                 		     		|
| Dropout       	    | dropout probability 0.5                    	|
| Fully connected      	| output 1                  		     		|


The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting I first added Dropout layers with a dropout probability of 0.5 after the fully connected layers and with a dropout probability of 0.3 after the convolutional layers. That was reducing overfitting pretty good but the performance of the model went down. So I changed it to only have dropout layers with probability 0.5 before each fully connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Creation of the Training Set & Training Process

First I started to create my own training data but the simulater in the workspace was really laggy. I stated a few attempts but it was really hard for me to steer the car nice and clean around the track. So in the end I decied to stick to the training data already given in the workspace.


At the beginning I started with just the images from the center cam. With the Nvidia netwok I could achieve a pretty good result for the irst corners with this data already. But if the car came too close to the edge of the lane it couln't really recover and get back to the lane center again.

So I also added the left and the right camera to the training data with an adjusted steering angle. In the end I chose 0.2 as correction factor for the steering which sometimes leads to a bit of oversterring an with that to lurching but the car was always pretty centered in the lane. In my first submission I used a correction factor of 0.1 which lead to a much smoother steering but in sharp curves the car came pretty close to the road edge which was rejected by the eviewer. With that data the car drove pretty good around the first corners even the sharp ones but when it arrived to the first right corner it couldn't take it properly.

To have the same amount of left and right corners in the training data I augmented the data by also adding the fipped version of all images with negative steering angles.

For training I shuffled the data set and put 20% of the data into a validation set to observe overfitting.


#### 5. Training process

After adding the dropout layers training loss and validation loss were both improving for the first 3 to 5 layers and then started overfitting. After trying training the model with different numbers of epochs I found out that the driving performance was best around 7 epochs. The model was then already a bit overfitted but that seemed to help in that case that the testing (driving with the simulater) was also on that distinct track. For driving in other enviroments/tracks thats for sure couterporductive. 

#### 6. Generator

I implemented a generator to read in date im my code. I used a batch size of 64 which lead to 64x6=384 images per batch. (Center, left, right image and everything flipped around)

### Possible further improvements

To generalize the model to also work on different tracks I would have to collect more data also from track 2. There would be also a lot of possibilities to augment data to abtain more different images.
One could also try out different model architectures and maybe try to tranfer learn with already pretrained CNNs which could be a bit difficult with that simulation enviroment with a bit artificial image quality.
