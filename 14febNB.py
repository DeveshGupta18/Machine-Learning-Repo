import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
data=load_iris()
#type(data)

print(data)
print(data.target_names)
print(data.data)

#######################
#seperating into dep and indep data
y=data.target
x=data.data
from sklearn.naive_bayes import GaussianNB
model1= GaussianNB()

#model fit on x, y dta
model1.fit(x,y)
######################
x_pred=[2,2,4,2]
output=model1.predict([x_pred,])
print("-"*50)
print("We have threee type of target : ", data.target_names)
print("Prediction is on : ", x_pred)
print("Prediction is : ", data.target_names[output])
print("-"*50)

def show(x_test):
    output=model1.predict([x_test,])
    print("-"*50)
    print("We have threee type of target : ", data.target_names)
    print("Prediction is on : ", x_test)
    print("Prediction is : ", data.target_names[output])
    print("-"*50)
    
sl=float(input("Enter Sepal length : "))
sw=float(input("Enter Sepal width : "))
pl=float(input("Enter Petal length : "))
pw=float(input("Enter Petal Width : "))
x_test = [sl, sw, pl, pw]
show(x_test)

################################
# USING MODEL2 SVM

y=data.target
x=data.data
from sklearn.svm import SVC
model2= SVC()

#model fit on x, y dta
model2.fit(x,y)

def show(x_test):
    output=model2.predict([x_test,])
    print("-"*50)
    print("We have threee type of target : ", data.target_names)
    print("Prediction is on : ", x_test)
    print("Prediction is : ", data.target_names[output])
    print("-"*50)
    
sl=float(input("Enter Sepal length : "))
sw=float(input("Enter Sepal width : "))
pl=float(input("Enter Petal length : "))
pw=float(input("Enter Petal Width : "))
x_test = [sl, sw, pl, pw]
show(x_test)

