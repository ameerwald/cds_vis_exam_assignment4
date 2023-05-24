import os
import pandas as pd
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt
# saving models 
from joblib import dump



def labels():
    # getting the label names from the train data
    classes = sorted(os.listdir("data/images/train"))
    # Remove '.DS_Store', something weird macs do
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')
    # Convert elements to lowercase
    classes = list(map(lambda x: x.lower(), classes))
    return classes

# loading the model 
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
    plt.savefig('out/plot_history.png')
    plt.show()

# function to generate additonal data
def data_generator():
    train_datagen = ImageDataGenerator(validation_split=0.2,
                                horizontal_flip=True, 
                                rotation_range=20,  
                                preprocessing_function = preprocess_input)
    test_datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,  
                                preprocessing_function = preprocess_input)
    return train_datagen, test_datagen


def train_model(train_datagen, test_datagen, classes, model): 
    BATCH_SIZE = 32
    TARGET_SIZE = (224,224)
    classes = ["apple", "avocado", "banana", "cherry", "kiwi", "mango", "orange", "pineapple", "strawberries", "watermelon"]
    # Training data
    train_images = train_datagen.flow_from_directory(
        "data/images/train",
        classes=classes,
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset="training",
        seed=42)
    # Validation data
    val_images = train_datagen.flow_from_directory(
        directory="data/images/train",
        classes=classes,
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset="validation",
        seed=42)
        # test data 
    test_images = test_datagen.flow_from_directory(
        "data/images/test",
        # fixing misspelled class label and making all classes lowercase 
        classes=list(map(lambda x:x.lower().replace("strawberries","stawberries"),classes)),
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        seed=42,
    )
    H = model.fit(train_images,
            batch_size=BATCH_SIZE,
            validation_data=val_images,
            steps_per_epoch=train_images.samples // BATCH_SIZE,
            validation_steps=val_images.samples // BATCH_SIZE,
            epochs=10,   
            verbose=1)
    predictions = model.predict(test_images, batch_size=32)
    return train_images, val_images, test_images, H, predictions




