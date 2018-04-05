from keras.models import Sequential
from keras.layers import Dense


import numpy as np
import pandas as pd
import csv

if __name__ == "__main__":
  # Fix the seed
    np.random.seed(40000)

    shuf = pd.read_csv("/Users/wahab/.spyder-py3/resources/test_data_1.csv")
    shuf = shuf.values
    cols = shuf.shape[1]
    rows = shuf.shape[0]
    shuf2 = shuf
    
    colss = ['mean_R','mean_G','mean_B','mean_H','mean_S','mean_I','max_mean_R','max_mean_G','max_mean_B','max_mean_H','max_mean_S','max_mean_I','min_mean_R','min_mean_G','min_mean_B','min_mean_H','min_mean_S','min_mean_I','max_R','max_G','max_B','max_H','max_S','max_I','min_R','min_G','min_B','min_H','min_S','min_I','label']
    print(colss)
    for i in range (cols-1): #bcz last column contains labels
        mean = np.mean(shuf2[:,i])
        std = np.std(shuf2[:,i],ddof=1)
        for j in range (rows):
            shuf2[j][i] = (shuf2[j][i] - mean)/(std)
            
    df_n = pd.DataFrame(np.array(shuf2), columns = colss)
    print (df_n)

    df_n.to_csv("/Users/wahab/.spyder-py3/resources/normalized_features_test_data_1.csv", index=False)        
    
    print("done")
    
    

