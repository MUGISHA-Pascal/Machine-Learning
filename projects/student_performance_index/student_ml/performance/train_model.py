import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train():
    df = pd.read_csv('dataset.csv')

    df['Extracurricular Activities']  = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    X = df.drop('Performance Index', axis = 1)
    y = df['Performance Index']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    model = RandomForestRegressor(n_estimators = 200, random_state = 42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'performance/model.pkl')
    print("Model trained and saved")