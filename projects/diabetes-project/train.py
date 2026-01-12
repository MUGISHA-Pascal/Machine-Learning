from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt

data_bunch=load_diabetes()
data=pd.DataFrame(data_bunch.data,columns=data_bunch.feature_names)
x=data_bunch.data
y=data_bunch.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeRegressor()
model.fit(x_train,y_train)
joblib.dump(model,"./model/diabetes.joblib")
y_pred=model.predict(x_test)
accuracy=r2_score(y_test,y_pred)
print(f"the accuracy is {accuracy}")

plt.figure(figsize=(15,10))
plot_tree(model,feature_names=data_bunch.feature_names,filled=True)
# plt.scatter(y_test,y_pred,color="red")
plt.title("prediction")
plt.show()
# print(data.head())