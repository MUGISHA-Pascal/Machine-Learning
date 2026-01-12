from flask import Flask,request,jsonify
import joblib

app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    features=data["features"]
    model=joblib.load("./model/videogame.joblib")
    prediction=model.predict([features])
    return jsonify({"prediction":prediction.tolist()})
if __name__=="__main__":
    app.run(port=5000,debug=True)
