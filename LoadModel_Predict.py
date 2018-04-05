#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:13:10 2018

@author: wahab
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import numpy as np
import os
import pandas as pd
import csv

if __name__ == "__main__":
    
    
    df1 = pd.read_csv("/Users/wahab/.spyder-py3/resources/normalized_features_test_data_1.csv")

    print(df1.shape)
    
    result = df1.values
    
    X = result[:,0:30]
    Y = result[:,30]
    
    
    # load json and create model
    json_file = open('/Users/wahab/.spyder-py3/model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/Users/wahab/.spyder-py3/model/model.h5")
    print("Loaded model from disk")
    
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    score = loaded_model.evaluate(X, Y)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    # calculate predictions
    predictions = loaded_model.predict(X)
    #round predictions
    rounded = [round(x[0]) for x in predictions]
    
    print(rounded)
    poscount=0
    for i in range (Y.shape[0]):
        if rounded[i]==1:
            poscount=poscount+1
    print(poscount)