import os
import keras
from keras.models import Sequential, save_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle

# seeding for reproducibility
np.random.seed(7)

# inputs and respective outputs
X = []
Y = []

# preprocessing training data
images = []
for img in os.listdir('images'):
    frame = cv2.imread(r'./images/{}'.format(img), cv2.IMREAD_GRAYSCALE)

    # adding correct classification for training data
    if 'perfect' in img:
        Y.append(0)
    elif 'thumbsup' in img:
        Y.append(1)
    elif 'peace' in img:
        Y.append(2)

    # image preprocessing
    frame = frame[95:600, 0:550]
    frame = cv2.Canny(frame, 100, 200)
    cv2.imshow('image', frame)
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    frame = frame[:, :, np.newaxis]

    images.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# convert all images to numpy arrays
X = np.array(images)

# shuffle classification data
X, Y = shuffle(X, Y, random_state=0)

# split into test and train set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

# input image dimensions
img_x, img_y = 252, 275
input_shape = (img_x, img_y, 1)

# convert class vectors to binary class matricies for use in catagorical_crossentropy loss below
classifications = 3
y_train = keras.utils.to_categorical(y_train, classifications)
y_test = keras.utils.to_categorical(y_test, classifications)

# CNN model
model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2), strides=(2, 2), activation='relu', input_shape=input_shape))
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(Conv2D(250, kernel_size=(2, 2), strides=(2, 2), activation='relu'))
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(classifications, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# tensorboard data callback
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# training on data
model.fit(x_train, y_train, batch_size=100, epochs=2, validation_data=(x_test, y_test), callbacks=[tbCallBack])

# save weights post training
model.save('weights_v3.h5')
