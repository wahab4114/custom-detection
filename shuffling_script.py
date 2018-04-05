from keras.models import Sequential
from keras.layers import Dense


import numpy as np
import pandas as pd
import csv



if __name__ == "__main__":

    # Fix the seed
    np.random.seed(40000)

    df1 = pd.read_csv("/Users/wahab/.spyder-py3/resources/training_data_pos1.csv")

    print(df1.shape)


    df2 = pd.read_csv("/Users/wahab/.spyder-py3/resources/training_data_neg1.csv")

    print(df2.shape)

    result = pd.concat([df1,df2])
    result = result.sample(frac=1)
    
    result.to_csv("/Users/wahab/.spyder-py3/resources/shuffled_data.csv", index=False)

    
   


  
