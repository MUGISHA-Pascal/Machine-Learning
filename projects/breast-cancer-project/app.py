from flask import Flask,request,jsonify
import joblib
import pandas as pd

app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    model=joblib.load("./model/cancer.joblib")
    data=request.json
    features=data["features"]
    input_data=pd.DataFrame([features],columns=["mean radius","mean texture"])
    prediction=model.predict(input_data)
    # prediction=model.predict([features])
    return jsonify({"prediction":prediction.tolist()})

if __name__ == "__main__":
    app.run(port=5000,debug=True)