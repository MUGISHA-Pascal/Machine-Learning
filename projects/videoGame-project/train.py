import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
import joblib
import numpy as np

data = pd.read_csv("data/vgsales.csv")
x=data.drop(["Name","Rank","Platform","Genre","Publisher","Year"],axis=1)
y=data["Rank"]
scale=StandardScaler()
x=scale.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy=r2_score(y_test,y_pred)
joblib.dump(model,"./model/videogame.joblib")

# print(f"accuracy is {accuracy}")
print(data.head())