import matplotlib.pyplot as plt
import joblib

model=joblib.load("../model/music.joblib")
features=["age","gender"]
feature_importance=model.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(features,feature_importance,color="skyblue")
plt.xlabel("feature importance")
plt.ylabel("features")
plt.title("music generation")
plt.show()