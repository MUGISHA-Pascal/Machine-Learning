from flask import Flask,request,jsonify
import joblib
import pandas as pd
from sklearn.datasets import load_wine

app=Flask(__name__)

@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    databunch=load_wine()
    features=data["features"]
    input_data=pd.DataFrame([features],columns=databunch.feature_names)
    model=joblib.load("./model/wine.joblib")
    prediction=model.predict(input_data)
    return  jsonify({"prediction":prediction.tolist()})
if __name__ == "__main__":
    app.run(port=5000,debug=True)