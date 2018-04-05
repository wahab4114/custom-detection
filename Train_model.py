from keras.models import Sequential
from keras.layers import Dense


import numpy as np
import pandas as pd
import csv



if __name__ == "__main__":

    
   
    df1 = pd.read_csv("/Users/wahab/.spyder-py3/resources/normalized_features.csv")

    print(df1.shape)
    
    result = df1.values
    
    X = result[:,0:30]
    Y = result[:,30]
    
    

    
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=100, batch_size=100)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("/Users/wahab/.spyder-py3/model/model.json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("/Users/wahab/.spyder-py3/model/model.h5")
    print("Saved model to disk")

    