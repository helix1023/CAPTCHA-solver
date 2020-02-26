#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress AVX Warning

def load_data():
    return tf.keras.datasets.mnist.load_data()

def shape_data(train, test):
    train = train.reshape(train.shape[0], 28, 28, 1)
    test = test.reshape(test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

    train = train.astype('float32')
    test = test.astype('float32')

    train /= 256
    train /= 256

    print('train shape:', train.shape)
    print('Number of images in train', train.shape[0])
    print('Number of images in test', train.shape[0])
    return input_shape, train, test

def generate_model(shape):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    return model

def save_model(model):
    model_json = model.to_json()

    with open("model/model.json", 'w') as json_file:
        json_file.write(model_json)

    model.save_weights("model/model.h5")

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape, x_train, x_test = shape_data(x_train, x_test)
    model = generate_model(input_shape)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=5)
    results = model.evaluate(x_test, y_test)

    save_model(model)

if __name__ == "__main__":
    main()
