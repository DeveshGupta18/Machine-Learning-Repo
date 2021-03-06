import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# k-fold ---------- cross validation
# gridsearchcv
# random forest
# bagging
# ensemble learning



from sklearn.datasets import load_iris
data=load_iris()


x= data.data
y=data.target

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test=train_test_split(x, y, test_size = 0.33, random_state = 42)

from sklearn.svm import SVC
model = SVC()
model.fit(x,y)
#model.fit(x_train, y_train)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(model, x, y, cv=10)
print(acc)

############################################

from sklearn.model_selection import GridSearchCV
par=[{'C' : [1,10,100,1000], 'kernel' : ['linear']},
      {'C' : [1,10,100,1000], 'kernel' : ['rbf'], 
       'gamma' : [0.1, 0.01,0.001, 0.0001]}]

GS = GridSearchCV(estimator = model, param_grid=par, scoring ='accuracy', 
                  cv = 2 , n_jobs=-1)
###########################
model_gs = GS.fit(x,y)
#model_gs1 = GS.fit(x_train, y_train)
##########################
print(model_gs.best_score_)
print(model_gs.best_params_)

#print(model_gs1.best_score_)
#print(model_gs1.best_params_)

#############################
# final model
#############################

opt_model= SVC(C=10, kernel = 'linear')
#opt_model= SVC(C=1, kernel = 'linear')
#preparing the data for the test
x_test = [2,2,4,2]

#prep opt model and fit to data
opt_model.fit(x,y)
#opt_model.fit(x_train,y_train)

#predicting value according to opt model
y_pred=opt_model.predict([x_test,])
#y_pred=opt_model.predict(x_test)
print(y_pred)
print(data.target_names[y_pred])

#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

#print("classification report is : \n",classification_report(y_test,y_pred))


#print("Accuracy of model  is : \n",accuracy_score(y_test,y_pred))

#print("Confusion matrix is  : \n",confusion_matrix(y_test,y_pred))


###################################################
# different model
from sklearn.tree import DecisionTreeClassifier
dt =DecisionTreeClassifier()

from sklearn.svm import SVC
svm_m=SVC()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.ensemble import VotingClassifier
model_en = VotingClassifier([("Decision_tree", dt), 
                             ("support_vectpr", svm_m),
                             ("KNN", knn)])

#fitting ensemble model to data
model_en.fit(x, y)

x_test=[2,2,4,2]
#predicting the value according to optmodel
y_pred = model_en.predict([x_test,])
print(model_en.score(x,y))
print(y_pred)
print(data.target_names[y_pred])

########
#Bagging ------------- Random Forest

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_bagging = BaggingClassifier(dt)
model_bagging.fit(x,y)
x_test = [2,2,4,2]

y_pred = model_bagging.predict([x_test, ])
print(model_bagging.score(x,y))
print(y_pred)
print(data.target_names[y_pred])

############################################
# random forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(x,y)
x_test=[2,2,4,2]
y_pred=model_rf.predict([x_test,])
print(model_rf.score(x,y))
print(y_pred)
print(data.target_names[y_pred])