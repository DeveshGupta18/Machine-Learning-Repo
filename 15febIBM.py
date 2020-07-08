import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\IBM HR.csv")
data.info()
data.isnull().sum()

data=data.drop(["EmployeeCount", "Over18", "EmployeeNumber","StandardHours"], axis=1)

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

'''
d=[]
for i in data.columns:
    d.append[i]
for i in d:
    data[i]=label_encoder.fit_transform(data[i])
'''

def cleaning(data):
    for i in data:
        if(str(data[i].dtype)=='object'):
            data[i]=label_encoder.fit_transform(data[i])
        
cleaning(data)
data.info()

###########################
y=data["Attrition"]
x=data.drop(["Attrition"], axis=1)

#########################

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,
                                                    random_state=42)

from sklearn.svm import SVC
model = SVC()

## model train and try to fit on training data
model.fit(x_train, y_train)


from sklearn.model_selection import cross_val_score
acc = cross_val_score(model, x, y, cv=10)
print(acc)

from sklearn.model_selection import GridSearchCV
par=[{'C' : [1,10,100,1000], 'kernel' : ['linear']},
      {'C' : [1,10,100,1000], 'kernel' : ['rbf'], 
       'gamma' : [0.1, 0.01,0.001, 0.0001]}]

    
GS = GridSearchCV(estimator = model, param_grid=par, scoring ='accuracy', 
                  cv = 10 , n_jobs=-1)
###########################
model_gs = GS.fit(x_train,y_train)
