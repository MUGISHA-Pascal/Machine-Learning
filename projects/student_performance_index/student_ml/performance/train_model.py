import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score
import numpy as np

def categorize_student(row):
    """
    Categorize students based on input features patterns
    Works with both pandas Series (with column names) and arrays/lists (with indices)
    """
    # Handle both DataFrame rows and arrays
    if isinstance(row, (list, np.ndarray)):
        # Array format: [hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]
        hours_studied = row[0]
        previous_scores = row[1]
        sleep_hours = row[3]
        sample_papers = row[4]
    else:
        # DataFrame row format
        hours_studied = row['Hours Studied']
        sleep_hours = row['Sleep Hours']
        previous_scores = row['Previous Scores']
        sample_papers = row['Sample Question Papers Practiced']
    
    # Extreme cases
    if hours_studied >= 9 and sleep_hours <= 4:
        return 'burnout_risk'
    elif sleep_hours >= 9 and hours_studied <= 2:
        return 'underachiever'
    elif hours_studied <= 2 and sleep_hours <= 4:
        return 'struggling'
    elif hours_studied >= 8 and previous_scores >= 85 and sample_papers >= 7:
        return 'high_performer'
    elif hours_studied >= 7 and sleep_hours >= 7 and sample_papers >= 5:
        return 'balanced_achiever'
    elif previous_scores >= 90 and hours_studied <= 4:
        return 'natural_talent'
    elif hours_studied >= 8 and previous_scores <= 50:
        return 'hard_worker'
    elif sleep_hours >= 8 and hours_studied >= 6 and sample_papers >= 4:
        return 'well_prepared'
    elif hours_studied <= 3 and previous_scores <= 50:
        return 'needs_support'
    else:
        return 'average_student'

def train():
    # Load dataset
    df = pd.read_csv('dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.describe())
    
    # Create student categories
    print("\n" + "="*60)
    print("CREATING STUDENT CATEGORIES")
    print("="*60)
    df['Student_Category'] = df.apply(categorize_student, axis=1)
    
    # Display category distribution
    print("\nCategory Distribution:")
    print(df['Student_Category'].value_counts())
    
    # Encode categorical variables
    label_encoder_activities = LabelEncoder()
    df['Extracurricular Activities'] = label_encoder_activities.fit_transform(df['Extracurricular Activities'])
    
    label_encoder_category = LabelEncoder()
    df['Student_Category_Encoded'] = label_encoder_category.fit_transform(df['Student_Category'])
    
    # Prepare features for both models
    X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y_regression = df['Performance Index']
    y_classification = df['Student_Category_Encoded']
    
    # Split data with stratification for classification
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    # Also create validation set
    X_train, X_val, y_reg_train, y_reg_val, y_class_train, y_class_val = train_test_split(
        X_train, y_reg_train, y_class_train, test_size=0.2, random_state=42, stratify=y_class_train
    )
    
    
    # ==================== TRAIN CLASSIFIER ====================
    print("\n" + "="*60)
    print("TRAINING CLASSIFIER MODEL")
    print("="*60)
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train, y_class_train)
    
    # Validation predictions for classifier
    y_class_val_pred = classifier.predict(X_val)
    val_class_accuracy = accuracy_score(y_class_val, y_class_val_pred)
    
    # Test predictions for classifier
    y_class_test_pred = classifier.predict(X_test)
    test_class_accuracy = accuracy_score(y_class_test, y_class_test_pred)
    
    print(f"\nClassifier Performance:")
    print(f"  Validation Accuracy: {val_class_accuracy:.4f}")
    print(f"  Test Accuracy:       {test_class_accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_class_test, y_class_test_pred, 
                                target_names=label_encoder_category.classes_))
    
    # ==================== TRAIN REGRESSOR ====================
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS")
    print("="*60)
    
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
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_reg_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        val_r2 = r2_score(y_reg_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_reg_val, y_val_pred))
        val_mae = mean_absolute_error(y_reg_val, y_val_pred)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_reg_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_test_pred))
        test_mae = mean_absolute_error(y_reg_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_reg_train, cv=5, scoring='r2')
        
        print(f"\n{name}:")
        print(f"  Validation R² Score: {val_r2:.4f}")
        print(f"  Validation RMSE:     {val_rmse:.4f}")
        print(f"  Validation MAE:      {val_mae:.4f}")
        print(f"  Test R² Score:       {test_r2:.4f}")
        print(f"  Test RMSE:           {test_rmse:.4f}")
        print(f"  Test MAE:            {test_mae:.4f}")
        print(f"  CV R² Score:         {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"  Feature Importances:")
            for feature, importance in zip(X.columns, model.feature_importances_):
                print(f"    {feature}: {importance:.4f}")
        
        # Track best model based on validation score
        if val_r2 > best_score:
            best_score = val_r2
            best_model = model
            best_name = name
    
    # Save both models and encoders
    model_package = {
        'classifier': classifier,
        'regressor': best_model,
        'label_encoder_category': label_encoder_category,
        'label_encoder_activities': label_encoder_activities
    }
    
    joblib.dump(model_package, 'performance/model.pkl')
    
    print("\n" + "="*60)
    print(f"BEST REGRESSION MODEL: {best_name}")
    print(f"Validation R² Score: {best_score:.4f}")
    print(f"Classifier Test Accuracy: {test_class_accuracy:.4f}")
    print("="*60)
    print("\nModels trained and saved to 'performance/model.pkl'")
    
    # Sample predictions with both models
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (Category + Score)")
    print("="*60)
    sample_data = [
        [7, 85, 1, 8, 3],  # Good student
        [7, 50, 1, 8, 3],  # Average student
        [3, 40, 0, 5, 1],  # Struggling student
        [9, 95, 1, 8, 9],  # Excellent student
        [9, 60, 0, 4, 2],  # Burnout risk
        [2, 90, 1, 9, 1],  # Natural talent
    ]
    
    sample_descriptions = [
        "Good student (7h study, 85 prev score, activities, 8h sleep, 3 papers)",
        "Average student (7h study, 50 prev score, activities, 8h sleep, 3 papers)",
        "Struggling student (3h study, 40 prev score, no activities, 5h sleep, 1 paper)",
        "Excellent student (9h study, 95 prev score, activities, 8h sleep, 9 papers)",
        "Burnout risk (9h study, 60 prev score, no activities, 4h sleep, 2 papers)",
        "Natural talent (2h study, 90 prev score, activities, 9h sleep, 1 paper)"
    ]
    
    for data, desc in zip(sample_data, sample_descriptions):
        category_pred = classifier.predict([data])[0]
        category_name = label_encoder_category.inverse_transform([category_pred])[0]
        score_pred = best_model.predict([data])[0]
        print(f"\n{desc}")
        print(f"  → Category: {category_name}")
        print(f"  → Predicted Performance Score: {score_pred:.2f}")