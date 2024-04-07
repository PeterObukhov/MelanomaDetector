import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.callbacks import *
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

class NeuralNet:
    metadata = "Dataset/HAM10000_metadata.csv"
    hmnist_8_8_RGB = "Dataset/hmnist_8_8_RGB.csv"
    hmnist_28_28_RGB = "backend/Dataset/hmnist_28_28_RGB.csv"
    hmnist_8_8_L = "Dataset/hmnist_8_8_L.csv"
    hmnist_28_28_L = "Dataset/hmnist_28_28_L.csv"

    classes = {4: ('nv', 'Меланоцитарный невус'),
            6: ('mel', 'Меланома'),
            2 :('bkl', 'Доброкачественное кератоподобное образование'), 
            1:('bcc' , 'Базально-клеточная карцинома'),
            5: ('vasc', 'Пиогенная гранулема'),
            0: ('akiec', 'Актинический кератоз'),
            3: ('df', 'Дерматофиброма')}

    def __init__(self):
        self.df = pd.read_csv(self.hmnist_28_28_RGB, delimiter=',')
        self.label = self.df["label"]
        self.data = self.df.drop(columns=["label"])
        if(os.path.isfile('backend/final.keras')):
            self.loadModel()
    
    def oversampleData(self):
        oversample = RandomOverSampler()
        self.data, self.label = oversample.fit_resample(self.data, self.label)
        self.data = np.array(self.data).reshape(-1,28,28,3)
        self.label = np.array(self.label)

    def trainModel(self):
        self.oversampleData()
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size = 0.2, random_state = 42)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 5, verbose=1, factor=0.5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
        self.model = self.createModel()
        self.model.compile(optimizer='adam', loss = keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
        self.model.fit(X_train, y_train, epochs=7, batch_size=128, validation_data=(X_test, y_test), callbacks=[learning_rate_reduction, early_stopping])
        
    def loadModel(self):
        self.model = keras.models.load_model('backend/final.keras')

    def predict(self, image):
        pred =  self.model.predict(image)
        return self.classes[pred[0].argmax(axis = 0)][1]

    def createModel():
        model = Sequential()
        #layer1
        model.add(Rescaling(1./255, input_shape=(28, 28, 3) ))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #layer2
        model.add(Conv2D(32, (3, 3) , padding='same'))
        model.add(Activation('relu'))
        #layer3
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        #layer4
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #classifier
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(7))
        model.add(Activation('softmax'))
        
        return model