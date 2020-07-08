
import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\sample loan prediction.csv")

data.head()

data.isnull().sum()
data.info()

data["Gender"].fillna(method="ffill",inplace=True)
data["Married"].fillna(method="ffill",inplace=True)
data["Dependents"].fillna(method="ffill",inplace=True)
data["Self_Employed"].fillna(method="ffill",inplace=True)
data.isnull().sum()

data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(),inplace=True)
data["Credit_History"].fillna(data["Credit_History"].median(),inplace=True)
data.isnull().sum()

data.pop("Loan_ID")

data.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Gender"]=le.fit_transform(data["Gender"])
data["Married"]=le.fit_transform(data["Married"])
data["Dependents"]=le.fit_transform(data["Dependents"])
data["Education"]=le.fit_transform(data["Education"])
data["Self_Employed"]=le.fit_transform(data["Self_Employed"])
data["Property_Area"]=le.fit_transform(data["Property_Area"])
data["Loan_Status"]=le.fit_transform(data["Loan_Status"])
data.info()

y=data["Loan_Status"]
x=data.drop(["Loan_Status"],axis=1)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5,metric='euclidean')

model.fit(x,y)
model.score(x,y)

from sklearn.svm import SVC
model=SVC()
model.fit(x,y)
model.score(x,y)



from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x,y)
model.score(x,y)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x,y)
model.score(x,y)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model.fit(x,y)
model.score(x,y)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
model.score(x,y)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x,y)