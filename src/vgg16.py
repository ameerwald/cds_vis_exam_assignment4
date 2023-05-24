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
from utils.preprocessing import load_model
from utils.preprocessing import plot_history
from utils.preprocessing import labels
from utils.preprocessing import data_generator
from utils.preprocessing import train_model
# saving models 
from joblib import dump



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
    with open(os.path.join("out", "classification_report.txt"), "w") as f:
        f.write(report)


if __name__=="__main__":
    main()