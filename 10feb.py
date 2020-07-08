###############################
#       LOGISTIC REGRESSION
##############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\churn data.csv")
data

###############################
#   DATA CLEANING
##############################
data.shape
data.columns
data.info()
########################
data.isnull().sum()

###############################
#
##############################
#direct removing
#removeed some irrelevent data
data_pre=data.iloc[0:, 4:]
data_pre.info()

#dummies label encoding
ip=pd.get_dummies(data_pre["international plan"])
ip.head()
ip.pop("no")

vp=pd.get_dummies(data_pre["voice mail plan"])
vp.head()
vp.pop("yes")
vp.head()

chu_red=pd.get_dummies(data_pre["churn"], drop_first=True)
chu_red.head()
##########################
data_p=pd.concat((data_pre, ip, vp, chu_red), axis=1)
data_p.shape
data_p.info()
final=data_p.drop(["churn", "voice mail plan", "international plan"], axis=1)
final.shape
final.info()

###################
#dividing into dep and independent 
y=final[True]
x=final.drop([True], axis=1)
x.shape
#############################
# split data into training and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size =0.25, random_state=20)

#TRAIN IS 75% AND TEST=25%
x_train.shape
x_train.info()
##########################
from sklearn.linear_model import LogisticRegression
#make a model on Logistic Regression
model=LogisticRegression()
#model fit data x_train, y_train to train it
model.fit(x_train, y_train)

#model test
y_pred=model.predict(x_test)

#import module to justify result
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification_report : \n", classification_report(y_test, y_pred))
print("Accuracy of model is : ", accuracy_score(y_test, y_pred))
print("Confusion matrix : " )
print(confusion_matrix(y_test, y_pred))

## accuracy
print((699+28)/(699+15+92+28))






