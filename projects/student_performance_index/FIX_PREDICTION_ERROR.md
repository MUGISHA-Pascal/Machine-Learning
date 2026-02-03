# Fix: "Unexpected token '<'" Error on Prediction

## What This Error Means
The error `Unexpected token '<', "<!DOCTYPE "... is not valid JSON` means:
- JavaScript is trying to parse JSON
- But the server is returning HTML (like an error page)
- This happens when the endpoint doesn't exist or has an error

## Quick Fix

### Step 1: Ensure Model is Trained
```bash
cd projects/student_performance_index/student_ml
python test_new_model.py
```

### Step 2: Restart Django Server
```bash
# Stop the server (Ctrl+C if running)
python manage.py runserver
```

### Step 3: Clear Browser Cache
- Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Or use Incognito/Private mode

### Step 4: Test the Endpoint
In a new terminal (while server is running):
```bash
python test_endpoint.py
```

## Detailed Troubleshooting

### Check 1: Is the Server Running?
```bash
# You should see:
# Starting development server at http://127.0.0.1:8000/
# Quit the server with CONTROL-C.
```

If not running, start it:
```bash
python manage.py runserver
```

### Check 2: Test Endpoint Manually
```bash
# While server is running, in another terminal:
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

**Expected Response (Good):**
```json
{
  "success": true,
  "student_category": "balanced_achiever",
  "predicted_performance_index": 78.5,
  "category_description": "Good study habits..."
}
```

**If you see HTML (Bad):**
```html
<!DOCTYPE html>
<html>
...
```
This means the endpoint has an error.

### Check 3: Look at Django Server Logs
When you make a prediction, check the terminal where Django is running.

**Good logs:**
```
[03/Feb/2026 10:30:45] "POST /predict-ajax/ HTTP/1.1" 200 156
```

**Bad logs (404 - endpoint not found):**
```
[03/Feb/2026 10:30:45] "POST /predict-ajax/ HTTP/1.1" 404 2345
```

**Bad logs (500 - server error):**
```
[03/Feb/2026 10:30:45] "POST /predict-ajax/ HTTP/1.1" 500 12345
Traceback (most recent call last):
  ...
```

### Check 4: Verify URLs Configuration
```bash
# Check if URLs are properly configured
python manage.py show_urls 2>/dev/null || python -c "
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
django.setup()
from django.urls import get_resolver
resolver = get_resolver()
for pattern in resolver.url_patterns:
    print(pattern)
"
```

You should see `/predict-ajax/` in the list.

### Check 5: Test in Browser Console
1. Open prediction page: `http://localhost:8000/predict-page/`
2. Press F12 to open DevTools
3. Go to Network tab
4. Make a prediction
5. Click on the `predict-ajax` request
6. Check the Response tab

**If you see HTML:** The endpoint has an error
**If you see JSON:** The endpoint works, but JavaScript has an issue

## Common Causes & Solutions

### Cause 1: Model Not Trained
**Symptom:** 500 error, logs show "Model file not found"

**Solution:**
```bash
python test_new_model.py
```

### Cause 2: Old Model Format
**Symptom:** 500 error, logs show "'RandomForestRegressor' object has no attribute 'classifier'"

**Solution:**
```bash
rm performance/model.pkl
python test_new_model.py
```

### Cause 3: URL Not Configured
**Symptom:** 404 error

**Solution:** Verify `performance/urls.py` has:
```python
path('predict-ajax/', predict_ajax, name='ajax_predict'),
```

### Cause 4: View Function Error
**Symptom:** 500 error with traceback in logs

**Solution:** Check the error in Django logs and fix the code

### Cause 5: CSRF Token Issue
**Symptom:** 403 Forbidden error

**Solution:** The view is already decorated with `@csrf_exempt`, but verify:
```python
@csrf_exempt
def predict_ajax(request):
    ...
```

### Cause 6: Wrong Content-Type
**Symptom:** Request fails silently

**Solution:** Already fixed in the updated code with proper headers

## Step-by-Step Debug Process

### 1. Check Server Status
```bash
# Is it running?
curl http://localhost:8000/
```

### 2. Check Endpoint Exists
```bash
# Does it return JSON or HTML?
curl -X POST http://localhost:8000/predict-ajax/ \
  -H "Content-Type: application/json" \
  -d '{"hours_studied": 8, "previous_scores": 85, "extracurricular": "true", "sleep_hours": 8, "sample_papers": 5}'
```

### 3. Check Model File
```bash
# Does it exist?
ls -lh performance/model.pkl

# Is it the new format?
python -c "
import joblib
m = joblib.load('performance/model.pkl')
print('Type:', type(m))
print('Is dict:', isinstance(m, dict))
if isinstance(m, dict):
    print('Keys:', m.keys())
"
```

### 4. Check Browser Console
- Open DevTools (F12)
- Console tab: Look for errors
- Network tab: Check request/response

### 5. Check Django Logs
- Look at terminal where `runserver` is running
- Check for errors or tracebacks

## Complete Reset (Nuclear Option)

If nothing works, do a complete reset:

```bash
# 1. Stop the server (Ctrl+C)

# 2. Delete old model
rm performance/model.pkl

# 3. Delete Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 4. Retrain model
python test_new_model.py

# 5. Restart server
python manage.py runserver

# 6. Clear browser cache (Ctrl+Shift+R)

# 7. Test again
```

## Verification Script

Create `verify_setup.sh`:
```bash
#!/bin/bash
echo "Verifying setup..."

# Check model exists
if [ -f "performance/model.pkl" ]; then
    echo "✓ Model file exists"
else
    echo "✗ Model file missing - run: python test_new_model.py"
    exit 1
fi

# Check model format
python -c "
import joblib
m = joblib.load('performance/model.pkl')
if isinstance(m, dict) and 'classifier' in m:
    print('✓ Model has correct format')
else:
    print('✗ Model has old format - delete and retrain')
    exit(1)
" || exit 1

# Check if server is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✓ Server is running"
else
    echo "✗ Server not running - run: python manage.py runserver"
    exit 1
fi

# Test endpoint
RESPONSE=$(curl -s -X POST http://localhost:8000/predict-ajax/ \
  -H "Content-Type: application/json" \
  -d '{"hours_studied": 8, "previous_scores": 85, "extracurricular": "true", "sleep_hours": 8, "sample_papers": 5}')

if echo "$RESPONSE" | grep -q "student_category"; then
    echo "✓ Endpoint returns correct JSON"
    echo "✓ All checks passed!"
else
    echo "✗ Endpoint not returning correct data"
    echo "Response: $RESPONSE"
    exit 1
fi
```

Run it:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

## Still Not Working?

1. **Check Python version:** Should be 3.8+
2. **Check Django version:** Should be compatible
3. **Check dependencies:** `pip install -r requirements.txt` (if exists)
4. **Check file permissions:** Make sure files are readable
5. **Try different browser:** Test in Chrome, Firefox, etc.
6. **Check firewall:** Make sure port 8000 is not blocked

## Get More Help

If still stuck, gather this information:
1. Django server logs (copy the error)
2. Browser console errors (F12 → Console)
3. Network tab response (F12 → Network → predict-ajax → Response)
4. Model file size: `ls -lh performance/model.pkl`
5. Python version: `python --version`
6. Django version: `python -c "import django; print(django.VERSION)"`
