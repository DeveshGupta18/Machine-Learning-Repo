###############################
#       LOGISTIC REGRESSION
##############################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\train.csv")
data.info()

#remove unwanted data
#removeed some irrelevent data

data=data.drop(["PassengerId", "Name", "Ticket","Cabin"], axis=1)
data.info()

###############################
#   DATA CLEANING
##############################
data["Age"].hist()
data["Age"].fillna(data["Age"].median(),inplace=True)

########################
data.isnull().sum()


#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Sex"]=label_encoder.fit_transform(data["Sex"])
#here we have two missing values so we have to fill them for label encoding 
data["Embarked"].fillna(method='ffill',inplace=True)
data["Embarked"]=label_encoder.fit_transform(data["Embarked"])
data.info()

x=data.drop(["Survived"],axis =1 )

y=data["Survived"]

###################################
# train teat split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=11)
x_train
######################## 
# applying logistic regresiom
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score



from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification report is : \n",classification_report(y_test,y_pred))


print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))

##################################
p=0
max=0;
t=.01
t1=t
while t<1.0:    
    THRESHOLD=t
    y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)
    t=t+.01
    p=accuracy_score(y_test,y_pred)
    if p > max:
        max=p
        t1=t
    
print(max)
print(t1 )





##########################################################
##########################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\train.csv")
data.info()

#remove unwanted data
#removeed some irrelevent data

data=data.drop(["PassengerId", "Name", "Ticket","Cabin"], axis=1)
data.info()

###############################
#   DATA CLEANING
##############################
data["Age"].hist()
data["Age"].fillna(data["Age"].median(),inplace=True)

########################
data.isnull().sum()


#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data["Sex"]=label_encoder.fit_transform(data["Sex"])

#here we have two missing values so we have to fill them for label encoding 

data["Embarked"].fillna(method='ffill',inplace=True)
data["Embarked"]=label_encoder.fit_transform(data["Embarked"])
data.info()

################################
#training
x_train=data.drop(["Survived"],axis =1 )
y_train=data["Survived"]

######################## 
# applying logistic regresiom
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


################################################
#                   for testing

data1=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\test.csv")
data1.info()

data1=data1.drop(["PassengerId", "Name", "Ticket","Cabin"], axis=1)
data1.info()

###############################
#   DATA CLEANING

data1["Age"].hist()
data1["Age"].fillna(data["Age"].median(),inplace=True)
data1["Fare"].hist()
data1["Fare"].fillna(data["Fare"].median(),inplace=True)


########################
data1.isnull().sum()
data1.info()

#label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data1["Sex"]=label_encoder.fit_transform(data1["Sex"])

data1["Embarked"]=label_encoder.fit_transform(data1["Embarked"])
data1.info()
x_test=data1
#now prediction

y_pred=model.predict(x_test)
###################
#total saved person
print(y_pred.sum())


############################setting threshold values

while t<1.0:
    
THRESHOLD=.50
y_pred=np.where(model.predict_proba(x_test)[:,1]>THRESHOLD,1,0)

print(y_pred.sum())