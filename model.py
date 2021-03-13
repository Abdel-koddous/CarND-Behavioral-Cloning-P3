print("Training the model with data loaded in batches using generators")

import csv
import cv2
import numpy as np

rows=[]
data_path = '../data/'

with open(data_path + 'driving_log.csv') as csvfile:

    csv_reader = csv.reader(csvfile)
    
    for row in csv_reader:
        
        rows.append(row)
    
    rows.pop(0)

print("-------------> File has ", len(rows), "  lines <------------------")


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(rows, test_size=0.2)

import sklearn
from math import ceil
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    print("Input set size ==>", num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(3):
                        source_path = batch_sample[i] # This data was generated on my local machine
                        filename = source_path.split('/')[-1]
                        new_path = data_path + "IMG/" + filename

                        img = cv2.imread(new_path)
                        images.append(img)

                    # steering values
                    correction = 0.2

                    center_steering = float( batch_sample[3] )
                    left_steering = center_steering + correction 
                    right_steering = center_steering - correction

                    angles.extend([ center_steering, left_steering, right_steering ])
                    #angles.append(center_steering)

            # Data Augmentation
            
            augmented_imgs, augmented_measurments = [], []
            for image, measurement in zip(images, angles):
                augmented_imgs.append( image )
                augmented_measurments.append( measurement )

                augmented_imgs.append( cv2.flip(image, 1) )
                augmented_measurments.append( -1.0*measurement )
                
            # trim image to only see section with road
            X_train = np.array(augmented_imgs)
            y_train = np.array(augmented_measurments)
            if offset == len(samples)//batch_size:
                print("X_train -->", X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

### Load images and steering measurements

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Let's train our data
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

model = Sequential()
### Data preprocessing
# Corpping
model.add(Cropping2D(cropping=( (70,25), (0,0) ), input_shape=(160, 320, 3) ) )
# Normalization
model.add( Lambda( lambda x: (x /255.0) - 0.5) )

### Nvidia Self Driving Car Pipeline
model.add( Conv2D( 24, (5, 5), strides=(2, 2), padding='valid', activation='relu') )
model.add( Conv2D( 36, (5, 5), strides=(2, 2), padding='valid', activation='relu') )
model.add( Conv2D( 48, (5, 5), strides=(2, 2), padding='valid', activation='relu') )
model.add( Conv2D( 64, (3, 3), strides=(1, 1), padding='valid', activation='relu') )
model.add( Conv2D( 64, (3, 3), strides=(1, 1), padding='valid', activation='relu') )
model.add( Dropout(0.4) )
model.add( Flatten() )
model.add( Dense(100) )
model.add( Dense(50) )
model.add( Dense(10) )
#model.add( Dropout(0.3) )
model.add( Dense(1) ) # Steering Angle is my output

model.compile( loss='mse', optimizer='adam' ) # A regeression network vs previously seen classification networks

model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=3, verbose=1)

model.save( 'model_nvidia_dropout04_3epochs.h5' )

