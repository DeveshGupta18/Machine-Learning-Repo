import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X=np.array([[185,72],[170,56],[168,60],[179,68],
            [182,72],[188,77],[180,71],[180,70],[183,84],
            [180,88],[180,67],[177,76]], dtype=int)

#algo
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model
y_kmeans = model.fit_predict(X)
print(y_kmeans)

#####################################################################

import numpy as np
import pandas as pd

#import data
data=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Mall_Customers.csv")
data.head()

#extract data in to np array form type
X=data.iloc[0:, [3,4]].values
#######################################################################
#applying Kmeans and find the value fo n_cluster for minimizing wcss
from sklearn.cluster import KMeans
wcss= []
for i in range(1,12):
    kmean=KMeans(n_clusters=i)
    kmean.fit_predict(X)
    wcss.append(kmean.inertia_)


#applying ELBOW METHOD
#to identify the value of clusters
plt.plot(range(1,12), wcss)
plt.xlabel("cluster value")
plt.ylabel("WCSS")
plt.title('cluster vs wcss')
plt.show()

##########################################################################
# n=5
#as per plot values of wcss reduce slowly offer a fix point 
# for now 5 so take it as n_cluster=5
model_final=KMeans(n_clusters=5)
model_final
y_pred=model_final.fit_predict(X)

res=pd.DataFrame({"prediction" : y_pred})
data1=pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Mall_Customers.csv")
val = data1.iloc[0:, [3,4]]
final_value = pd.concat((val, res), axis=1)
