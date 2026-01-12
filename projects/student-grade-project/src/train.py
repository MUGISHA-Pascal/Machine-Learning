from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv("../data/student-mat.csv",sep=";")
data=data[["age","absences","G1","G2","G3"]]
x=data.drop("G3",axis=1)
y=data["G3"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
model=LinearRegression()
model.fit(x_train,y_train)
with open("../model/student.pickle","wb") as f:
    pickle.dump(model,f)
y_pred=model.predict(x_test)
accuracy=model.score(x_test,y_test)
m2=mean_squared_error(y_test,y_pred)
print(f"accuracy : {accuracy} , mean error : {m2}")
