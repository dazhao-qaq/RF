import pandas as pd
import numpy as np

dataset = pd.read_excel('数据整合.xlsx')

print(dataset.head())

data1 = dataset.loc[0, "meanNTL"]
data2 = dataset.loc[0, "sumNTL"]

for i in range(len(dataset)):
    data1 = dataset.loc[i, "meanNTL"]
    data2 = dataset.loc[i, "sumNTL"]
    print(data1, data2)
    if(i == 0) :
        continue
    else :
        if(data1 == dataset.loc[i-1, "meanNTL"]):
            print("ERR")
            exit(-1)
