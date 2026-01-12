import pandas as pd
import joblib
import argparse


def predict(features):
    # Load the trained model
    model = joblib.load('../models/iris_model.joblib')

    # Create a DataFrame for the input features
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    input_data = pd.DataFrame([features], columns=feature_names)

    # Make a prediction
    prediction = model.predict(input_data)

    return prediction[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the species of an Iris flower.')
    parser.add_argument('sepal_length', type=float, help='Sepal length')
    parser.add_argument('sepal_width', type=float, help='Sepal width')
    parser.add_argument('petal_length', type=float, help='Petal length')
    parser.add_argument('petal_width', type=float, help='Petal width')

    args = parser.parse_args()
    features = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]

    prediction = predict(features)
    print(f'The predicted species is: {prediction}')
