# Student Performance Index - Updated Model

## Overview
The student performance prediction model has been enhanced with a dual-model architecture that provides both **classification** and **regression** predictions.

## Key Updates

### 1. Classification Model
A Random Forest Classifier categorizes students into 10 distinct categories based on their study patterns:

#### Student Categories:
- **burnout_risk**: High study hours (≥9h) with insufficient sleep (≤4h)
- **underachiever**: Excessive sleep (≥9h) with minimal study time (≤2h)
- **struggling**: Low study hours (≤2h) and insufficient sleep (≤4h)
- **high_performer**: Excellent balance (≥8h study, ≥85 prev score, ≥7 papers)
- **balanced_achiever**: Good study habits (≥7h study, ≥7h sleep, ≥5 papers)
- **natural_talent**: High scores (≥90) with minimal study effort (≤4h)
- **hard_worker**: High effort (≥8h study) despite lower previous scores (≤50)
- **well_prepared**: Good balance (≥8h sleep, ≥6h study, ≥4 papers)
- **needs_support**: Low engagement (≤3h study, ≤50 prev score)
- **average_student**: Typical student profile (default category)

### 2. Regression Model
Predicts the numerical performance index (0-100) using the best performing model (Random Forest or Gradient Boosting).

### 3. Validation During Training
- **Train/Validation/Test Split**: Data is split into training (64%), validation (16%), and test (20%) sets
- **Validation Monitoring**: Model performance is monitored on validation set during training
- **Best Model Selection**: The regression model with the best validation R² score is selected
- **Stratified Sampling**: Classification uses stratified sampling to maintain category distribution

### 4. Structured Output
API responses now include:
```json
{
  "student_category": "balanced_achiever",
  "predicted_performance_index": 78.5,
  "category_description": "Good study habits with adequate sleep and preparation"
}
```

## Model Architecture

### Training Pipeline
1. **Data Loading**: Load dataset with student features
2. **Category Creation**: Apply categorization rules to create labels
3. **Data Encoding**: Encode categorical variables
4. **Data Splitting**: Split into train/validation/test sets with stratification
5. **Classifier Training**: Train Random Forest Classifier
6. **Classifier Validation**: Evaluate on validation and test sets
7. **Regressor Training**: Train multiple regression models
8. **Regressor Validation**: Monitor performance on validation set
9. **Model Selection**: Select best regressor based on validation R²
10. **Model Packaging**: Save both models with encoders

### Model Package Structure
```python
{
    'classifier': RandomForestClassifier,
    'regressor': RandomForestRegressor or GradientBoostingRegressor,
    'label_encoder_category': LabelEncoder for categories,
    'label_encoder_activities': LabelEncoder for extracurricular activities
}
```

## Usage

### Training the Model
```bash
cd projects/student_performance_index/student_ml
python test_new_model.py
```

Or via Django shell:
```bash
python manage.py shell
>>> from performance.train_model import train
>>> train()
```

### Making Predictions

#### Via API (POST /api/predict/):
```python
import requests

data = {
    'hours_studied': 7,
    'previous_scores': 85,
    'extracurricular': True,
    'sleep_hours': 8,
    'sample_papers': 5
}

response = requests.post('http://localhost:8000/api/predict/', json=data)
result = response.json()

print(f"Category: {result['student_category']}")
print(f"Score: {result['predicted_performance_index']}")
print(f"Description: {result['category_description']}")
```

#### Via Python:
```python
import joblib
import numpy as np

# Load model
model_package = joblib.load('performance/model.pkl')

# Prepare features
features = np.array([[7, 85, 1, 8, 5]])  # [hours, prev_score, activities, sleep, papers]

# Get predictions
category = model_package['classifier'].predict(features)[0]
category_name = model_package['label_encoder_category'].inverse_transform([category])[0]
score = model_package['regressor'].predict(features)[0]

print(f"Category: {category_name}, Score: {score:.2f}")
```

## Training Output

The training process now displays:
1. **Dataset Information**: Shape and statistics
2. **Category Distribution**: Count of students in each category
3. **Classifier Performance**: 
   - Validation accuracy
   - Test accuracy
   - Classification report with precision/recall/F1
4. **Regressor Performance** (for each model):
   - Validation R², RMSE, MAE
   - Test R², RMSE, MAE
   - Cross-validation scores
   - Feature importances
5. **Sample Predictions**: Examples showing both category and score

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-category performance metrics

### Regression Metrics
- **R² Score**: Coefficient of determination (validation and test)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **CV Score**: 5-fold cross-validation R² score

## Files Modified

1. **train_model.py**: 
   - Added `categorize_student()` function
   - Implemented dual-model training pipeline
   - Added validation monitoring
   - Enhanced output with both predictions

2. **views.py**:
   - Updated `load_model()` to handle model package
   - Modified `predict_performance()` to return both predictions
   - Added `get_category_description()` helper function
   - Updated all AJAX endpoints for new structure

3. **test_new_model.py** (new):
   - Comprehensive test script
   - Validates both classification and regression
   - Tests various student profiles

## Benefits

1. **Richer Insights**: Provides both categorical and numerical predictions
2. **Better Monitoring**: Validation set helps prevent overfitting
3. **Interpretability**: Categories provide human-readable student profiles
4. **Actionable**: Category descriptions suggest intervention strategies
5. **Robust**: Stratified sampling ensures balanced training

## Example Predictions

| Student Profile | Category | Score | Description |
|----------------|----------|-------|-------------|
| 8h study, 90 score, activities, 8h sleep, 7 papers | high_performer | 85.2 | Excellent balance |
| 9h study, 70 score, no activities, 4h sleep | burnout_risk | 68.5 | Risk of burnout |
| 2h study, 95 score, activities, 9h sleep | natural_talent | 82.3 | High scores, minimal effort |
| 2h study, 40 score, no activities, 4h sleep | struggling | 28.7 | Needs support |
| 9h study, 45 score, activities, 7h sleep, 8 papers | hard_worker | 52.4 | High effort |

## Next Steps

Consider these enhancements:
1. Add confidence scores for predictions
2. Implement feature importance visualization
3. Create category-specific recommendations
4. Add time-series tracking for student progress
5. Implement ensemble methods for improved accuracy
