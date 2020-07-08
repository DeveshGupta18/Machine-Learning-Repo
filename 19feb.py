import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

data=pd.read_csv(r"C:\Users\Bharat Gupta\Downloads\Mall_Customers.csv")
x=data.iloc[0:,[3,4]]
dendo=sch.dendrogram(sch.linkage(x,method='ward')) #ward minimizes the Varience within cluster and KMeans++ minimize WCSS
plt.title('Dendrogram')
plt.xlabel('CUstomers')
plt.ylabel("ED")
plt.show()  

#############################################

from sklearn.cluster import hierarchical
model=hierarchical.AgglomerativeClustering(n_clusters=4 )
y_sch=model.fit_predict(x)

###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns

data=pd.read_csv(r"C:\Users\Bharat Gupta\Desktop\Edge Download\Wine.csv")
x.info()

# data['Malic_Acid']=np.where(data["Malic_Acid"]>5,5,data["Malic_Acid"])
# sns.boxplot(data['Malic_Acid'])
# x=data.iloc[0:,0:-1]
# data['Ash']=np.where(data['Ash']>3.00,3.00,data['Ash'])
# data['Ash']=np.where(data['Ash']<1.75,1.75,data['Ash'])
# sns.boxplot(data['Ash'])
# data['Ash_Alcanity']=np.where(data['Ash_Alcanity']>27.5,27.5,data['Ash_Alcanity'])
# data['Ash_Alcanity']=np.where(data['Ash_Alcanity']<11,11,data['Ash_Alcanity'])
# data['Magnesium']=np.where(data['Magnesium']>135,135,data['Magnesium'])
# data['Proanthocyanins']=np.where(data['Proanthocyanins']>3.0,3.0,data['Proanthocyanins'])
# data['Color_Intensity']=np.where(data['Color_Intensity']>10,10,data['Color_Intensity'])
# data['Hue']=np.where(data['Hue']>1.45,1.45,data['Hue'])
# sns.boxplot(data['Proline'])





dendo=sch.dendrogram(sch.linkage(x,method='ward')) #ward minimizes the Varience within cluster and KMeans++ minimize WCSS
plt.title('Dendrogram')
plt.xlabel('CUstomers')
plt.ylabel("ED")
plt.show()

###################################################

d=pd.read_csv(r"C:\Users\Bharat Gupta\Desktop\Edge Download\Wine.csv")
x=d.drop(['Customer_Segment'],axis=1)
y=d['Customer_Segment']
from sklearn.neighbors import KNeighborsClassifier
modelknn=KNeighborsClassifier()
modelknn.fit(x,y)
print(modelknn.score(x,y))
x_test=[13.49,1.66,2.24,24,87,1.88,1.84,0.27,1.03,3.74,0.98,2.78,472]
y_pred=modelknn.predict([x_test])
print(y_pred)
y_pro=modelknn.predict_proba([x_test])
print(y_pro)

# cross validation--score
from sklearn.model_selection import cross_val_score

acc=cross_val_score(modelknn,x,y,cv=10)
print(acc)

from sklearn.model_selection import GridSearchCV
# par=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.01,0.001,0.0001]}]
par=[{'n_neighbors':[3,5,7,9,11]}]

gs=GridSearchCV(estimator=modelknn, param_grid=par,scoring="accuracy",cv=10,n_jobs=-1)

# gs.fit(x,y)
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=19)

model_gs=gs.fit(x,y)
print(model_gs.best_score_,"\n")
print(model_gs.best_params_)

# optimized model
print('\n Optimized Model ---- \n')
modelknn_opt=KNeighborsClassifier(n_neighbors=3)
modelknn_opt.fit(x,y)
print(modelknn_opt.score(x,y))
x_test=[13.49,1.66,2.24,24,87,1.88,1.84,0.27,1.03,3.74,0.98,2.78,472]
y_pred=modelknn_opt.predict([x_test])
print(y_pred)
y_pro=modelknn_opt.predict_proba([x_test])
print(y_pro)

#############################################################

d=pd.read_csv(r"C:\Users\Bharat Gupta\Desktop\Edge Download\Wine.csv")
x=d.drop(['Customer_Segment'],axis=1)
y=d['Customer_Segment']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
print(model.score(x,y))
x_test=[13.49,1.66,2.24,24,87,1.88,1.84,0.27,1.03,3.74,0.98,2.78,472]
y_pred=model.predict([x_test])
print(y_pred)
y_pro=model.predict_proba([x_test])
print(y_pro)