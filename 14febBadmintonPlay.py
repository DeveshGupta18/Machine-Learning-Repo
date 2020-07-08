import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv(r"C:\Users\DELL\Downloads\Compressed\Machine-Learning-Tutorial-master\data\id3.csv")
data

data.isnull().sum()
data.info()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

d=[]
for i in data.columns:
    d.append(i)
for i in d:
    data[i]=label_encoder.fit_transform(data[i])

y=data['Answer']
x=data.drop(['Answer'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x,y)

'''

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x, y)

out = int(input(print("Enter outlook: " )))
temp = int(input(print("Enter temperature: " )))
hum = int(input(print("Enter Humidity: " )))
wind = int(input(print("Enter Wind: " )))
x_test=[out, temp, hum, wind]
y_pred = model.predict(x_test)
list = ["No", "Yes"]
print(list[int(x_test)])
'''



from sklearn.metrics import accuracy_score
print("accuracy : " , accuracy_score(y_test, y_pred))