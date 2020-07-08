import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\churn data.csv")
data.info()

data=data.iloc[0:, 4:]
data.isnull().sum()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

d=[]
for i in data.columns:
    d.append(i)
for i in d:
    data[i]=le.fit_transform(data[i])
    
y=data["churn"]
x=data.drop(["churn"], axis=1)

