import keras
from keras.preprocessing.image import ImageDataGenerator
from squeezenet import SqueezeNet
import numpy as np
import os
import pickle
import cv2

batch_size = 32
num_classes = 3
epochs = 50


def load_data(path, new_size, num_img):
    '''
    read in the images and resize them
    '''
    print("read in images...")
    X = np.empty([num_img, new_size[0], new_size[1], 3])
    for i in range(num_img):
        img = cv2.imread(os.path.join(path, str(i)+".jpg"))
        X[i] = cv2.resize(img, new_size[::-1]) #cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print()
    print("finished")
    return X

path = './carla.hal-master/ros/src/tl_detector/imgs'
y_train = np.loadtxt(os.path.join(path, "label.txt"))
x_train= load_data(path, (100,75), len(y_train))


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = x_train.astype(np.float32)
x_train /= 255. 

img_shape = x_train[0].shape
model = SqueezeNet(img_shape, num_classes)


# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
	vertical_flip=False) # randomly flip images
datagen.fit(x_train)



#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
			                        steps_per_epoch=x_train.shape[0] // batch_size,
			                        epochs=epochs)

model.save(os.path.join(path, 'model.h5'))