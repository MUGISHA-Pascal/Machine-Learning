from flask import Flask,request,jsonify
import joblib
import pandas as pd

app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    model=joblib.load("./model/diabetes.joblib")
    data=request.json
    features=data["features"]
    inputdata=pd.DataFrame([features],columns=["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"])
    prediction=model.predict(inputdata)
    return jsonify({"prediction":prediction.tolist()})
if __name__=="__main__":
    app.run(port=5000,debug=True)
