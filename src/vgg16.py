# import packages - basics 
import pandas as pd
import os
import numpy as np
import cv2
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.metrics import classification_report
# for plotting
import matplotlib.pyplot as plt
# importing functions from preprocessing script 
import sys
sys.path.append(".")
# import my functions in the utils folder 
from utils.preprocessing import labels
from utils.preprocessing import data_generator
from utils.preprocessing import train_model
# saving models 
from joblib import dump

# loading the model with extra layers 
def load_model(classes):
    model = VGG16()
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(32, 32, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)          
    class1 = Dense(256,                         
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(len(classes), 
                activation='softmax')(class2)
    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig('out/vgg16_plot_history.png')
    plt.show()

def main():
    # get labels for the data
    classes = labels()
    # load the model
    model = load_model(classes)
    # generate extra data 
    train_datagen, test_datagen = data_generator()
    # train the model
    train_images, val_images, test_images, H, predictions = train_model(train_datagen, test_datagen, classes, model)
    # Plot history over the epochs 
    plot_history(H, 10)
    # classification report 
    report = (classification_report(test_images.classes,
                                predictions.argmax(axis=1),
                                target_names=classes))
    # save the report 
    with open(os.path.join("out", "vgg16_classification_report.txt"), "w") as f:
        f.write(report)


if __name__=="__main__":
    main()