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

#total row 1470, train 67% & test=33%
##model Decision Tree

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

## model train and try to fit on training data
model.fit(x_train, y_train)

### model testing on test data

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy is : " ,accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))

from sklearn import tree
with open(r"C:\Users\DELL\Desktop\ML Files\model_tree.txt","w") as f:
    f=tree.export_graphviz(model, out_file=f)

################
    #now view the decision tree using webgraphviz.com
    
