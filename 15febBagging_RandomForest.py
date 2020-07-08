########
#Bagging ------------- Random Forest

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_bagging = BaggingClassifier(dt)
model_bagging.fit(x,y)
x_test = [2,2,4,2]

y_pred = model_en.predict([x_test, ])
print(y_pred)
print(data.target_names[y_pred])