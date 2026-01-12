import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

data_bunch = load_breast_cancer()
data = pd.DataFrame(data_bunch.data, columns=data_bunch.feature_names)
data=data[["mean radius","mean texture","worst fractal dimension"]]
x=data.drop("worst fractal dimension",axis=1)
y=data["worst fractal dimension"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,"./model/cancer.joblib")
y_pred=model.predict(x_test)
# meansq=mean_squared_error(y_test,y_pred)
# print(f"the mean square error is : {meansq}")
# variance_score=explained_variance_score(y_test,y_pred)
# print(f"the variance score is : {variance_score}")
#
# plt.scatter(y_test,y_pred)
# plt.xlabel("x features")
# plt.ylabel("y features")
# plt.title("the analysis")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
# plt.show()
#
prediction=model.predict([[17.24,18.43]])
print(prediction)