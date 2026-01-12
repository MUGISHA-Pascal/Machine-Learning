from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data=pd.read_csv("../data/music.csv")
x=data.drop("genre",axis=1)
y=data["genre"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
joblib.dump(model,"../model/music.joblib")
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy  : {accuracy}")
