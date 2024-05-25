#Importing the Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten, Lambda
from keras.applications.vgg16 import preprocess_input 
from keras.models import Sequential
from glob import glob 

#resize all the images to this size 
IMAGE_SIZE = [224,224]

train_path = r"C:\Users\DELL\Downloads\archive (37)\train"
test_path = r"C:\Users\DELL\Downloads\archive (37)\test"

#add VGG
vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False
    
import glob

folders = glob.glob(r"C:\Users\DELL\Downloads\archive (37)\train\*")

#Add layers 
x = Flatten()(vgg.output)
x = Dense(500, activation='relu')(x)
predictions = Dense(len(folders), activation = 'softmax')(x)

#create the model object 
model = Model(inputs = vgg.input, outputs = predictions)

model.summary()

#Compilation process
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



#PREPROCESS THE DATASET 

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')


testing_set = test_datagen.flow_from_directory(test_path,
                                                 target_size = (224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')

#TRAINING THE MODEL 
history = model.fit(training_set, 
                    validation_data=testing_set,
                    epochs = 15)
















