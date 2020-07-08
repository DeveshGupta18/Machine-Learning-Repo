
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Social_Network_Ads.csv")
data.info()

#remove unwanted data
#removeed some irrelevent data

data=data.drop(["User ID"], axis=1)
data.info()

data.isnull().sum()
###############################
#   DATA CLEANING
##############################
data["Age"].hist()
data["Age"].fillna(data["Age"].mean(),inplace=True)
data["EstimatedSalary"].hist()
data["EstimatedSalary"].fillna(data["EstimatedSalary"].mean(),inplace=True)
########################
data.isnull().sum()

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Gender"]=label_encoder.fit_transform(data["Gender"])


x=data.drop(["Purchased"],axis =1 )

y=data["Purchased"]

###################################
# train teat split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=10)
x_train
######################## 

# model 1 = Logistic Regression
# applying logistic regresiom

from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()

model1.fit(x_train,y_train)

y_pred=model1.predict(x_test)


#model 2 = KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

## using minkowski distance matrix

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train, y_train)

THRESHOLD = 0.50
import numpy as mp
y_pred = np.where(model2.predict_proba(x_test)[:,1] > THRESHOLD, 1, 0)
print(y_pred.sum())
print(y_pred)

#model 3 = SVC
from sklearn.svm import SVC
model3 = SVC()
model3.fit(x_train, y_train)
#y_pred = model3.predict(x_test)
#print(y_pred)

y_label = ["Not Purchased", "Purchased"]

def show(x_test):
    output=model3.predict([x_test,])
    print("-"*50)
    print("We have two type of target : ", y_label)
    print("Prediction is on : ", x_test)
    print("Prediction is : ", y_label[int(output)])
    print("-"*50)
    
sl=float(input("Enter Gender : "))
sw=float(input("Enter Age : "))
pl=float(input("Enter Estimated Salary : "))
x_test = [sl, sw, pl]
show(x_test)





from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))

