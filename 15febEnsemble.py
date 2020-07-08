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

