from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import joblib

california = fetch_california_housing()
# print(california)
data = pd.DataFrame(california.data,columns=california.feature_names)
data["MedHouseVal"] = california.target
print(data.head())

# print(data.isnull().sum())

x = data.drop("MedHouseVal",axis=1)
y=data["MedHouseVal"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# print(f"Training data shape : {x_train.shape}")
# print(f"Training data shape : {x_test.shape}")

model = LinearRegression()
model.fit(x_train,y_train)
#
# print("model trained successfully")
joblib.dump(model,"./model/california.joblib")
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

# print(f"mean squared error : {mse}")
# print(f"R-squared : {r2}")