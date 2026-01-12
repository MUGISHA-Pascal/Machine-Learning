from sklearn.datasets import load_wine
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree,DecisionTreeClassifier
import matplotlib.pyplot as plt

data_bunch=load_wine()
data=pd.DataFrame(data_bunch.data,columns=data_bunch.feature_names)
x=data_bunch.data
y=data_bunch.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=10,random_state=42)
model.fit(x_train,y_train)
joblib.dump(model,"./model/wine.joblib")
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
# print(f"accuracy is {accuracy}")
# tree=model.estimators_[0]
# plt.figure(figsize=(10,6))
# plot_tree(tree,feature_names=data_bunch.feature_names,class_names=data_bunch.target_names,filled=True)
# plt.show()
print(data_bunch.feature_names)