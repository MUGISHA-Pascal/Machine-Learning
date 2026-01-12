import pandas as pd
from flask import Flask,request,jsonify
import joblib

app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    features=data["features"]
    feature_name=["age","gender"]
    input_data=pd.DataFrame([features],columns=feature_name)
    model=joblib.load("./model/music.joblib")
    print(features)
    prediction=model.predict(input_data)
    return jsonify({"prediction":prediction.tolist()})
if __name__=="__main__":
    app.run(port=5000,debug=True)
