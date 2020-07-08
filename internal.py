import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#loading iris dataset
from sklearn.datasets import load_iris
data=load_iris()

#seperating dependent variable and independent variable
x= data.data
y=data.target

#seprating datatset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=11)


#using Support Vector Machine
#using Support Vector Classifier from sklearn.svm
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

plt.hist(y_pred)