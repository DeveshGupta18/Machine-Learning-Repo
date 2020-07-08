import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\DELL\Downloads\Regression.csv")
data.info()

#########################
#   Data Cleaning
#########################
data.isnull().sum()
#age is having some null values 
data['Age'].plot.hist()
#the histogram is not equally distributed so we can't use mean
#but we can use median
data['Age'].fillna(data['Age'].median(), inplace=True)
data.isnull().sum()
data.info()

#to convert object types in numeric for the model we will create their dummies
#convert to dummies
#rule is n-1
jt=pd.get_dummies(data["Job Type"],drop_first=True)
jt.head()

Ms=pd.get_dummies(data["Marital Status"],drop_first=True)
Ms.head()


Edu=pd.get_dummies(data["Education"])#,drop_first=True)
Edu.pop("Secondry")
Edu.head()


city=pd.get_dummies(data["Metro City"])#,drop_first=True)
city.pop("Yes")
city.head()

#concat all dummies to my data
pre_final=pd.concat((data,jt,Ms,Edu,city),axis=1)
pre_final

#now removing unwanted columns
#data drop
final_data=pre_final.drop(["Metro City","Education","Marital Status","Job Type"],axis=1)


#outlier treatment
import seaborn as sns
#checking the outlier
sns.boxplot(final_data["Age"])

final_data["Age"]=np.where(final_data["Age"]>60,60,final_data["Age"])
#treated box plt
sns.boxplot(final_data["Age"])


#final_data.info()

sns.boxplot(final_data['Signed in since(Days)'])
final_data['Signed in since(Days)']=np.where(final_data['Signed in since(Days)']<45,45, final_data['Signed in since(Days)'])
sns.boxplot(final_data['Signed in since(Days)'])
#############################################

# sep dependent and independent
y=final_data["Purchase made"] #dep data
x=final_data.drop(["Purchase made"],axis=1) #independent data
#######################################################3


###########################
#OPTIMIZING AND BACKWARD ELIMINATION

import statsmodels.api as sm
X=np.append(arr=np.ones((325,1)).astype(int), values=x, axis=1)
print(X)

x_opt=X[0:, [0,1,2,3,4,5,6,7,8,9]]
#x_opt.shape
model1=sm.OLS(endog=y, exog=x_opt).fit()
model1.summary()
###################
#backwark eliminaiton
x_opt2=X[0:, [0,1,2,3,4,5,6,9]]
model2=sm.OLS(endog=y, exog=x_opt2).fit()
model2.summary()

x_opt3=X[0:, [0,1,2,6,9]]
model3=sm.OLS(endog=y, exog=x_opt3).fit()
model3.summary()

x_opt4=X[0:, [0,2,6,9]]
model4=sm.OLS(endog=y, exog=x_opt4).fit()
model4.summary()

#########################
#   Data Modeling
#########################

#data drop
x=x.drop(["Age","Retired","Student","Unemployed", "Graduate", "Primary"],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train.shape
y_train.shape
x_test.shape
#x_test.to_csv(r"C:\Users\fluXcapacit0r\Desktop\test.csv")
#x_t=pd.read_csv(r"C:\Users\fluXcapacit0r\Desktop\test.csv")
####################################################
#model
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#model train
model.fit(x_train,y_train)

##########################
# DATA TESTING
#########################
#predict value of purchase made
y_pred=model.predict(x_test)
#summary 
####################################


#############################
#   R2_SCORE
#############################
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)




