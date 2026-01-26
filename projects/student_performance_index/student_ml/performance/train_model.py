import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def train():
    # Load dataset
    df = pd.read_csv('dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.describe())
    
    # Encode categorical variable
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    
    # Prepare features and target
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Try multiple models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    best_model = None
    best_score = -float('inf')
    best_name = None
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        print(f"\n{name}:")
        print(f"  Train R² Score: {train_r2:.4f}")
        print(f"  Test R² Score:  {test_r2:.4f}")
        print(f"  Test RMSE:      {test_rmse:.4f}")
        print(f"  Test MAE:       {test_mae:.4f}")
        print(f"  CV R² Score:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"  Feature Importances:")
            for feature, importance in zip(X.columns, model.feature_importances_):
                print(f"    {feature}: {importance:.4f}")
        
        # Track best model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_name = name
    
    # Save best model
    joblib.dump(best_model, 'performance/model.pkl')
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_name}")
    print(f"Test R² Score: {best_score:.4f}")
    print("="*60)
    print("\nModel trained and saved to 'performance/model.pkl'")
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    sample_data = [
        [7, 85, 1, 8, 3],  # Good student
        [7, 50, 1, 8, 3],  # Average student
        [3, 40, 0, 5, 1],  # Struggling student
        [9, 95, 1, 8, 9],  # Excellent student
    ]
    
    for data in sample_data:
        pred = best_model.predict([data])[0]
        print(f"Input: {data} -> Predicted Performance: {pred:.2f}")