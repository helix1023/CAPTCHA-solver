#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import random
import json
import os
from train import load_data, shape_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress AVX Warning

def print_example(test, model, r):
    plt.style.use("dark_background")
    fig = plt.figure()
    data = test[r].reshape(28, 28)
    plt.imshow(data)
    pred = model.predict(test[r].reshape(1, 28, 28, 1))
    title = 'Prediction: ' + str(pred.argmax())
    plt.title(title)
    plt.show()

def print_n_examples(test, model, n):
    for i in range(n):
        random_i = int(random.random() * 9999)
        print_example(test, model, random_i)

# Due to a bug where softmax gets serialized as softmax_v2
def fix_serialize(json_model_str):
    json_model = json.loads(json_model_str)

    for layer in json_model["config"]["layers"]:
        if "activation" in layer["config"].keys():
            if layer["config"]["activation"] == "softmax_v2":
                layer["config"]["activation"] = "softmax"

    return json.dumps(json_model)

def load_model():
    with open("model/model.json", 'r') as json_file:
        json_model = json_file.read()

    json_model = fix_serialize(json_model)

    model = model_from_json(json_model)

    model.load_weights("model/model.h5")  
    print("Loaded Model")
    return model

def main():
    model = load_model()
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape, x_train, x_test = shape_data(x_train, x_test)
    print_n_examples(x_test, model, 10)

if __name__ == "__main__":
    main()
