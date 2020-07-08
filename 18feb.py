import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Mall_Customers.csv")

data.info()
data.isnull().sum()
######################
#   DATA CLEANING
######################
data=data.drop(["CustomerID"], axis=1)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

d=[]
for i in data.columns:
    d.append(i)
for i in d:
    data[i]=label_encoder.fit_transform(data[i])

y=data["Spending Score (1-100)"]
x=data.drop(["Spending Score (1-100)"], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size = 0.33, random_state=42)


from sklearn.linear_model import LinearRegression
model=LinearRegression()

#model train
model.fit(x_train,y_train)

#predict value of purchase made
y_pred=model.predict(x_test)
#summary 
####################################

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
