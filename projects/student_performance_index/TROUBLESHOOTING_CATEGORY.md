# Troubleshooting: Category Not Showing on Prediction Page

## Problem
When making predictions on the web interface, the category information is not displayed.

## Most Likely Cause
**The model was trained with the OLD version before the category feature was added.**

## Solution: Retrain the Model

### Step 1: Delete Old Model
```bash
cd projects/student_performance_index/student_ml
rm performance/model.pkl
```

### Step 2: Retrain with New Version
```bash
# Option A: Using test script (Recommended)
python test_new_model.py

# Option B: Using Django shell
python manage.py shell
>>> from performance.train_model import train
>>> train()
>>> exit()

# Option C: Using the web interface
# 1. Start server: python manage.py runserver
# 2. Go to: http://localhost:8000/train/
# 3. Click "Train Model"
```

### Step 3: Verify Model Structure
```bash
python test_prediction_fix.py
```

Expected output:
```
✓ Model loaded successfully
  - Classifier: RandomForestClassifier
  - Regressor: RandomForestRegressor
  - Categories: 10 categories

✓ Test Case 1: Balanced Student
  → Category: Balanced Achiever
  → Performance Score: 78.50
```

### Step 4: Restart Django Server
```bash
# Stop the server (Ctrl+C)
# Start it again
python manage.py runserver
```

### Step 5: Clear Browser Cache
- Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Or open browser DevTools (F12) → Network tab → Check "Disable cache"

## Debugging Steps

### Check 1: Verify Model File Exists
```bash
ls -lh performance/model.pkl
```

If the file is very small (< 1MB), it's likely the old version.

### Check 2: Test Model in Python
```python
import joblib
import numpy as np

# Load model
model_package = joblib.load('performance/model.pkl')

# Check structure
print("Keys in model package:", model_package.keys() if isinstance(model_package, dict) else "Not a dict - OLD MODEL!")

# If it's a dict, check components
if isinstance(model_package, dict):
    print("Has classifier:", 'classifier' in model_package)
    print("Has regressor:", 'regressor' in model_package)
    print("Has label_encoder_category:", 'label_encoder_category' in model_package)
```

Expected output:
```
Keys in model package: dict_keys(['classifier', 'regressor', 'label_encoder_category', 'label_encoder_activities'])
Has classifier: True
Has regressor: True
Has label_encoder_category: True
```

If you see "Not a dict - OLD MODEL!", you need to retrain.

### Check 3: Test API Endpoint
```bash
# Start server
python manage.py runserver

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/predict-ajax/ \
  -H "Content-Type: application/json" \
  -d '{
    "hours_studied": 8,
    "previous_scores": 85,
    "extracurricular": "true",
    "sleep_hours": 8,
    "sample_papers": 5
  }'
```

Expected response:
```json
{
  "success": true,
  "student_category": "balanced_achiever",
  "predicted_performance_index": 78.5,
  "category_description": "Good study habits with adequate sleep and preparation"
}
```

If you see only `predicted_performance_index` without category, the model needs retraining.

### Check 4: Browser Console
1. Open the prediction page
2. Press F12 to open DevTools
3. Go to Console tab
4. Make a prediction
5. Look for the log: `API Response: {...}`

Check if the response includes:
- `student_category`
- `category_description`
- `predicted_performance_index`

## Common Issues

### Issue 1: "Model not found" Error
**Solution:** Train the model first
```bash
python test_new_model.py
```

### Issue 2: Category Shows "Not Available"
**Cause:** Old model file without classifier
**Solution:** Delete `performance/model.pkl` and retrain

### Issue 3: Error: "only integers, slices... are valid indices"
**Cause:** Fixed in the updated code
**Solution:** 
1. Make sure you have the latest `train_model.py` and `views.py`
2. Retrain the model
3. Restart the server

### Issue 4: Category Shows but Wrong Color
**Cause:** JavaScript color mapping issue
**Solution:** Clear browser cache (Ctrl+Shift+R)

### Issue 5: 500 Internal Server Error
**Cause:** Model structure mismatch
**Solution:**
1. Check Django logs in terminal
2. Delete old model: `rm performance/model.pkl`
3. Retrain: `python test_new_model.py`
4. Restart server

## Quick Fix Script

Create a file `fix_category_issue.sh`:
```bash
#!/bin/bash
echo "Fixing category display issue..."

# Navigate to project
cd projects/student_performance_index/student_ml

# Backup old model (optional)
if [ -f "performance/model.pkl" ]; then
    echo "Backing up old model..."
    mv performance/model.pkl performance/model.pkl.backup
fi

# Retrain model
echo "Retraining model with category support..."
python test_new_model.py

# Test predictions
echo "Testing predictions..."
python test_prediction_fix.py

echo "Done! Restart your Django server and refresh the browser."
```

Run it:
```bash
chmod +x fix_category_issue.sh
./fix_category_issue.sh
```

## Verification Checklist

After retraining, verify:
- [ ] `test_new_model.py` runs without errors
- [ ] `test_prediction_fix.py` shows categories
- [ ] Browser console shows category in API response
- [ ] Prediction page displays category badge
- [ ] Category description appears
- [ ] Category badge has correct color
- [ ] Performance score still shows correctly

## Still Not Working?

### Check File Versions
Make sure you have the updated files:

**train_model.py** should have:
```python
def categorize_student(row):
    """
    Categorize students based on input features patterns
    Works with both pandas Series (with column names) and arrays/lists (with indices)
    """
    # Handle both DataFrame rows and arrays
    if isinstance(row, (list, np.ndarray)):
        ...
```

**views.py** should have:
```python
def get_category_description(category):
    """Provide human-readable descriptions for categories"""
    descriptions = {
        'burnout_risk': 'High study hours with insufficient sleep - risk of burnout',
        ...
```

**predict.html** should have:
```html
<div class="category-badge" id="studentCategory">--</div>
<div class="category-description" id="categoryDescription">--</div>
```

### Get Help
If still not working:
1. Check Django server logs for errors
2. Check browser console for JavaScript errors
3. Verify all files are saved
4. Try in a different browser
5. Check if model file size is > 1MB (indicates new version)

## Prevention
To avoid this issue in the future:
1. Always retrain after code updates
2. Version your model files (e.g., `model_v2.pkl`)
3. Add model version checking in code
4. Document when model retraining is needed
