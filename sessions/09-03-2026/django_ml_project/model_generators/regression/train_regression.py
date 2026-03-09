import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Ensure we're in the right directory
dataset_path = "dummy-data/vehicles_ml_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

df = pd.read_csv(dataset_path)
features = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
target = "selling_price"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save model
joblib.dump(model, "model_generators/regression/regression_model.pkl")
# Predict
predictions = model.predict(X_test)
# Calculate R2 Score
r2 = round(r2_score(y_test, predictions) * 100, 2)
# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
 {
 "Actual": y_test.values,
 "Predicted": predictions.round(2),
 "Difference": (y_test.values - predictions).round(2),
 }
)
def evaluate_regression_model():
 return {
 "r2": r2,
 "comparison": comparison_df.head(10).to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 ),
 }
