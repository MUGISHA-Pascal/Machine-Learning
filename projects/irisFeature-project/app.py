from flask import Flask,request,jsonify
import joblib

app = Flask(__name__)
model = joblib.load('./irisdataset-project/models/iris_model.joblib')
@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    features=data["features"]
    prediction=model.predict([features])
    return jsonify({"prediction":prediction.tolist()})

if __name__ == "__main__":
    app.run(port=5000)