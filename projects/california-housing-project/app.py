from flask import Flask,request,jsonify
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

app=Flask(__name__)

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json
    features=data["features"]
    california=fetch_california_housing()
    model=joblib.load("./model/california.joblib")
    feature_names =california.feature_names
    input_data=pd.DataFrame([features],columns=feature_names)
    prediction=model.predict(input_data)
    return jsonify({"prediction":prediction.tolist()})

if __name__ == "__main__":
    app.run(port=5000,debug=True)