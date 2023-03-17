#!/usr/bin/env python
# coding: utf-8

# # changing to header after running it once
# import os, random
# import shutil
# dire = "C:/Users/lexus/Desktop/DeepLearning/cars_type/Train/"
# output = "C:/Users/lexus/Desktop/DeepLearning/cars_type/Test/"
# filename = ["Cab/","Convertible/","Coupe/", "Hatchback/", "Minivan/", "Other/", "Sedan/", "SUV/", "Van/", "Wagon/"]
# for i in filename:
#     file = os.listdir(dire+i)
#     test = random.sample(file, len(file)//4)
#     for j in test:
#         source = dire + i+j
#         dest = output + i
#         if os.path.isfile(source):
#             shutil.move(source, dest)
# 
#             

# In[1]:


import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

    


# In[2]:


#loading the models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input


# In[3]:


def create_model():
    def add_conv_block(model, no_filters):
        #deifining the arch of convulation layers
        model.add(Conv2D(no_filters, kernel_size=(3,3), activation = "relu")) #first we keep padding as same, as image size is already less
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(no_filters,kernel_size=(3,3), activation="relu" ))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(no_filters,kernel_size=(3,3), activation="relu" ))
        model.add(MaxPool2D(pool_size=(2,2)))
        
        return model
    model = Sequential()
    model.add(Input(shape = (255,255,3))) #adding input layer with image shape as (32,32,3)
    model = add_conv_block(model, 128) #running the function with 32 kernels 1st
    
    
    #next we flatten the image and add dense layer
    model.add(Flatten())
    model.add(Dense(128,activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")
    return model


# In[4]:


car_classifier = create_model()
car_classifier.summary()


# In[7]:


#defining the image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train = ImageDataGenerator(rescale = 1/255., rotation_range= 45, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, horizontal_flip= True, fill_mode="reflect")
test = ImageDataGenerator(rescale = 1/255.)


# In[3]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
keras_callbacks = [EarlyStopping(monitor = "val_loss", mode = "min", min_delta = 0.01, patience = 10), ModelCheckpoint("./car_best_weights", monitor="val_loss", mode = "min",save_best_only=True)]


# In[19]:


train_data = train.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Train",target_size=(255,255),batch_size= 32,class_mode="categorical")
test_data =  test.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Test",target_size=(255,255),batch_size= 32,class_mode="categorical")


# In[21]:


car_classifier.fit(train_data, epochs = 20,steps_per_epoch=train_data.samples//32, validation_data=test_data, validation_steps=test_data.samples//32, callbacks = keras_callbacks)


# #using transfer learning to improve model performnce

# In[4]:


base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top= False) #include top = False is given to ignore loading fully connected netwrok, as we will train the model with custom Deep learning Netwrok, and only use convulational layers from here
base_model.summary()


# In[5]:


base_model.trainable = False #to make sure that the convulation layers doesnt learn any new weights


# In[8]:


train2 = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input, rotation_range= 45, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, horizontal_flip= True, fill_mode="reflect")
test2 = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)


# In[9]:


train_data2 = train2.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Train",target_size=(224,224),batch_size= 64,class_mode="categorical")
test_data2 =  test2.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Test",target_size=(224,224),batch_size= 64,class_mode="categorical")


# In[26]:


transfer_model = Sequential([base_model, Flatten(), Dense(units = 128 , activation = "relu"),BatchNormalization(),Dense(units = 16 , activation = "relu"),BatchNormalization(),Dense(units = 10, activation = "softmax")])


# In[27]:


transfer_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")


# In[28]:


transfer_model.fit(train_data2, epochs = 17,steps_per_epoch=train_data2.samples//64, validation_data=test_data2, validation_steps=test_data2.samples//64, callbacks = keras_callbacks)


# In[13]:


base_model2 =tf.keras.applications.NASNetMobile(input_shape=(224,224,3), include_top= False)
base_model2.summary()


# In[14]:


base_model2.trainable = False


# In[15]:


train3 = ImageDataGenerator(preprocessing_function=tf.keras.applications.nasnet.preprocess_input, rotation_range= 45, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, horizontal_flip= True, fill_mode="reflect")
test3 = ImageDataGenerator(preprocessing_function=tf.keras.applications.nasnet.preprocess_input)


# In[20]:


train_data3 = train3.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Train",target_size=(224,224),batch_size= 64,class_mode="categorical")
test_data3 =  test3.flow_from_directory("C:/Users/lexus/Desktop/DeepLearning/cars_type/Test",target_size=(224,224),batch_size= 64,class_mode="categorical")


# In[21]:


nasnet_model = Sequential([base_model2, Flatten(), Dense(units = 256 , activation = "relu"),BatchNormalization(),Dense(units = 32 , activation = "relu"),BatchNormalization(),Dense(units = 10, activation = "softmax")])


# In[22]:


nasnet_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")


# In[24]:


nasnet_model.fit(train_data3, epochs = 20,steps_per_epoch=train_data3.samples//64, validation_data=test_data3, validation_steps=test_data3.samples//64, callbacks = keras_callbacks)


# In[34]:


#testing the images
class_names = train_data2.class_indices
class_names


# In[29]:


#predicting with CNN model
images = [] #we create an empty list, then apend it with all our test images, after resizing them
file = "C:/Users/lexus/Desktop/DeepLearning/cars_type/Examples/"
filename = os.listdir(file)
for i in filename:
    image = cv2.imread(file+i)
    img = cv2.resize(image, (255,255))
    images.append(img)

images_ar = np.array(images) #next we convert that list to an array, and scale the images
images = images_ar/255.


# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')
preds = car_classfier.predict(images)
names = np.array(list(class_names.keys()))
for j in range(len(preds)):
    pos = np.array(list(class_names.values())) == preds[j].argmax()
    img = cv2.imread(file+filename[j])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print(names[pos])


# In[31]:


#predicting with transfer learning model
images = [] #we create an empty list, then apend it with all our test images, after resizing them
file = "C:/Users/lexus/Desktop/DeepLearning/cars_type/Examples/"
filename = os.listdir(file)
for i in filename:
    image = cv2.imread(file+i)
    img = cv2.resize(image, (224,224))
    images.append(img)

images_ar = np.array(images) #next we convert that list to an array, and scale the images
images = tf.keras.applications.mobilenet_v2.preprocess_input(images_ar)


# In[35]:


pred = transfer_model.predict(images)
pred.round()
names = np.array(list(class_names.keys()))
names


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
for j in range(len(pred)):
    pos = np.array(list(class_names.values())) == pred[j].argmax()
    img = cv2.imread(file+filename[j])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print(names[pos])


# In[37]:


#predicting with NasNet learning model
images = [] #we create an empty list, then apend it with all our test images, after resizing them
file = "C:/Users/lexus/Desktop/DeepLearning/cars_type/Examples/"
filename = os.listdir(file)
for i in filename:
    image = cv2.imread(file+i)
    img = cv2.resize(image, (224,224))
    images.append(img)

images_ar = np.array(images) #next we convert that list to an array, and scale the images
images = tf.keras.applications.nasnet.preprocess_input(images_ar)


# In[38]:


ypred = nasnet_model.predict(images)
names = np.array(list(class_names.keys()))
names


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
for j in range(len(ypred)):
    pos = np.array(list(class_names.values())) == ypred[j].argmax()
    img = cv2.imread(file+filename[j])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print(names[pos])


# In[ ]:




