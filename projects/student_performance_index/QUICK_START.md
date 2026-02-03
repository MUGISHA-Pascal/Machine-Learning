# Quick Start Guide - Updated Student Performance Model

## Installation & Setup

1. **Navigate to project directory**:
```bash
cd projects/student_performance_index/student_ml
```

2. **Activate virtual environment** (if not already active):
```bash
source ../venv/bin/activate  # Linux/Mac
# or
..\venv\Scripts\activate  # Windows
```

3. **Install dependencies** (if needed):
```bash
pip install django djangorestframework scikit-learn pandas numpy joblib
```

## Training the Model

### Option 1: Using Test Script (Recommended)
```bash
python test_new_model.py
```

This will:
- Train both classifier and regressor
- Show detailed metrics
- Display sample predictions
- Validate the model works correctly

### Option 2: Using Django Shell
```bash
python manage.py shell
```
```python
from performance.train_model import train
train()
```

### Option 3: Via Web Interface
```bash
# Start the server
python manage.py runserver

# Navigate to: http://localhost:8000/train/
# Click "Train Model" button
```

## Making Predictions

### Via Python Script
```python
import joblib
import numpy as np

# Load the model
model_package = joblib.load('performance/model.pkl')

# Example: Balanced high performer
features = np.array([[8, 90, 1, 8, 7]])
# Format: [hours_studied, previous_scores, extracurricular(0/1), sleep_hours, sample_papers]

# Get predictions
category = model_package['classifier'].predict(features)[0]
category_name = model_package['label_encoder_category'].inverse_transform([category])[0]
score = model_package['regressor'].predict(features)[0]

print(f"Student Category: {category_name}")
print(f"Performance Score: {score:.2f}")
```

### Via API
```bash
# Start server
python manage.py runserver

# In another terminal, make a request:
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "hours_studied": 8,
    "previous_scores": 90,
    "extracurricular": true,
    "sleep_hours": 8,
    "sample_papers": 7
  }'
```

Expected response:
```json
{
  "student_category": "high_performer",
  "predicted_performance_index": 85.2,
  "category_description": "Excellent balance of study, preparation, and past performance"
}
```

### Via Web Interface
```bash
# Start server
python manage.py runserver

# Navigate to: http://localhost:8000/predict/
# Fill in the form and click "Predict"
```

## Understanding the Output

### Student Categories

| Category | Meaning | Typical Pattern |
|----------|---------|-----------------|
| **high_performer** | Top student | High study, high scores, well-prepared |
| **balanced_achiever** | Good student | Balanced study, sleep, and practice |
| **natural_talent** | Gifted student | High scores with minimal effort |
| **hard_worker** | Determined student | High effort despite lower past scores |
| **well_prepared** | Organized student | Good balance across all factors |
| **burnout_risk** | ⚠️ At risk | Too much study, too little sleep |
| **underachiever** | ⚠️ Needs motivation | Too much sleep, too little study |
| **struggling** | ⚠️ Needs help | Low study and sleep |
| **needs_support** | ⚠️ Critical | Low engagement overall |
| **average_student** | Typical student | Standard profile |

### Performance Score
- Range: 0-100
- Represents predicted academic performance index
- Higher scores indicate better expected performance

## Example Test Cases

### Test Case 1: High Performer
```python
Input: [8, 90, 1, 8, 7]
# 8 hours study, 90 previous score, has activities, 8 hours sleep, 7 practice papers

Expected:
- Category: high_performer or balanced_achiever
- Score: 80-95
```

### Test Case 2: Burnout Risk
```python
Input: [9, 70, 0, 4, 3]
# 9 hours study, 70 previous score, no activities, 4 hours sleep, 3 practice papers

Expected:
- Category: burnout_risk
- Score: 60-75
```

### Test Case 3: Natural Talent
```python
Input: [2, 95, 1, 9, 1]
# 2 hours study, 95 previous score, has activities, 9 hours sleep, 1 practice paper

Expected:
- Category: natural_talent
- Score: 75-90
```

### Test Case 4: Struggling Student
```python
Input: [2, 40, 0, 4, 1]
# 2 hours study, 40 previous score, no activities, 4 hours sleep, 1 practice paper

Expected:
- Category: struggling or needs_support
- Score: 20-35
```

## Troubleshooting

### Model file not found
```bash
# Train the model first
python test_new_model.py
```

### Import errors
```bash
# Install missing packages
pip install scikit-learn pandas numpy joblib
```

### Django errors
```bash
# Run migrations
python manage.py migrate

# Check if server is running
python manage.py runserver
```

## Validation Metrics

After training, you should see:

**Classifier:**
- Validation Accuracy: ~0.85-0.95
- Test Accuracy: ~0.85-0.95

**Regressor:**
- Validation R²: ~0.95-0.99
- Test R²: ~0.95-0.99
- RMSE: ~2-5
- MAE: ~1-3

If metrics are significantly lower, consider:
1. Checking data quality
2. Adjusting model hyperparameters
3. Adding more training data

## Next Steps

1. ✅ Train the model using `test_new_model.py`
2. ✅ Verify predictions with sample data
3. ✅ Start the Django server
4. ✅ Test the API endpoints
5. ✅ Integrate with your application

For detailed information, see `UPDATED_MODEL_README.md`
