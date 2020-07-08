#PCA ANALYSIS
#PCA is used to overcome the problem of overfitting
#PCA is used to reduce the higher dimension
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Wine.csv")
data

data.isnull().sum()

#.values for nd array
x=data.iloc[0:, 0:13].values
y=data.iloc[0:, 13:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, 
                                                    random_state=42)

#scale
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

opt_com=pca.explained_variance_
#ratio help to find the optimum value
opt_com_ratio = pca.explained_variance_ratio_


#from the ration we can conclude the value of n_components =2

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

#model test
y_pred=model.predict(x_test)


from sklearn.metrics import accuracy_score
print("Accuracy = ", accuracy_score(y_test, y_pred))

