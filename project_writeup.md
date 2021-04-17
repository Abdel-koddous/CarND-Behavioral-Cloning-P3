# **Behavioral Cloning Project** 
#### Abdelkododus Khamsi


[image1]: ./writeup_images/nvidia_nn.JPG "Nvidia End to End Learning NN"
[image2]: ./writeup_images/rawHistogram.JPG "Udacity raw driving log"
[image3]: ./writeup_images/flattenedHistogram.JPG "Udacity flattened driving log"

[gif1]: ./writeup_images/driveAroundTheTrack.gif "Final model driving around the track"

The goal of this project is to drive a car around a track by generating steering commands as out of Neural Network pipleine where the input is the front cameras images of the vehicle.

The main steps I went through when working on this project are:
* Gather various training data from the simulator.
* Choose and implement the model pipeline architecture.
* Finetune the implementation parameters.

## Rubric implementation points:

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

* At first I tried using the basic LeNet Architecture but I soon switched to ended a model based on Nvidia End to End learning NN introduced in this paper [EndToEndLearningForSelf-DrivingCars](EndToEndLearningForSelf-DrivingCars.pdf).

  ![alt text][image1]

#### 2. Attempts to reduce overfitting in the model
* I added a dropout layer after the last convolutional layer of the NN as it seemed to handle better the various turn curves (especially the ones with white/red marking) after this modification. The value of the dropout I setteled for after manual finetuning is 0.4
* Used a 80/20 training/validation split.
#### 3. Model parameter tuning
* An Adam Optimizer was used for the model compilation.
#### 4. Appropriate training data

Initially I tried with my own data:
    * Used straightforward and inverse laps.
    * Portions where the focus is to take the turns/curves smoothly.
    * Portions where I do off track recovery (record on the way back).
    * 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

* At first I tried using the basic LeNet Architecture but ended up using a model Based on Nvidia End to End learning NN introduced in this paper [EndToEndLearningForSelf-DrivingCars].

* I then changed the activation function to relu and added a dropout layer after the last fc layer of the pipeline.

* Used a 80/20 training/validation split, used ONE drop out layer after all convolutions, Try after each one of the conv ...
* Used adam optimizer (an alternative of the Stochastic gradient descent)
#### 2. Final Model Architecture
* I added a dropout layer after the last convolutional layer of the NN as it seemed to handle better the curves with this modification.
* My end model is handeling better the curves with red/white marking vs the ones with yellow/gray marking. A potential option I could look into is to feed the training data with more data where the car recovers from an almost off the track position around yellow/grey marking.

#### 3. Creation of the Training Set & Training Process

* Even if in my end model I used exclusively the dataset provided by Udacity I went through the following steps to make the most out of it:
* Flipping the images and taking the opposite value of the steering angle.
* Used right and left camera images and introduced a correction factor of 0.2 to the steering angles.


## Update

After receiving the review on my project I made a modification on the distribution of the udacity driving log by filtering out around 75% of the images that have a near zero (`0.05`) steering angle. This way I was able to flatten the distribution and reduce the bias of my model around zero steering angle.

  ![alt text][image2]

  ![alt text][image3]
  
### Demo time!

Updated model driving around the track:

  ![alt text][gif1]

The full video can be found [here](2laps20mphAccelerated3.mp4).
