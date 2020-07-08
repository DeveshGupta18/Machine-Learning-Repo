#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
model_lr=LinearRegression()
model_lr.fit(x_train,y_train)
y_pred=model_lr.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(x_train, y_train)
y_pred=model_LR.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("classification_report : \n", classification_report(y_test, y_pred))
print("Accuracy of model is : ", accuracy_score(y_test, y_pred))
print("Confusion matrix : " )
print(confusion_matrix(y_test, y_pred))


#KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train, y_train)
y_pred=model_knn.predict(x_test)

print("classification_report : \n", classification_report(y_test, y_pred))
print("Accuracy of model is : ", accuracy_score(y_test, y_pred))
print("Confusion matrix : " )
print(confusion_matrix(y_test, y_pred))


#SVC CLASSIFIER

from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(x_train, y_train)
y_pred=model_svc.predict(x_test)

print("classification_report : \n", classification_report(y_test, y_pred))
print("Accuracy of model is : ", accuracy_score(y_test, y_pred))
print("Confusion matrix : " )
print(confusion_matrix(y_test, y_pred))


#DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
y_pred = model_dt.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy is : " ,accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
from sklearn import tree
with open(r"C:\Users\DELL\Desktop\ML Files\model_tree.txt","w") as f:
    f=tree.export_graphviz(model, out_file=f)

