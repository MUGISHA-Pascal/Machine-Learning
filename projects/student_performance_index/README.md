# Student Performance Prediction System

A machine learning-powered Django web application that predicts student performance using a dual-model approach combining classification and regression to provide comprehensive insights into student academic patterns.

## Overview

This project implements an advanced predictive system to estimate student performance using machine learning. The system analyzes factors such as study hours, previous scores, extracurricular activities, sleep patterns, and practice question papers to:
1. **Classify students** into 10 distinct behavioral categories
2. **Predict performance index** (0-100 scale) using regression

## Key Features

### Dual-Model Architecture
- **Classification Model**: Random Forest Classifier categorizes students into 10 behavioral patterns
- **Regression Model**: Random Forest/Gradient Boosting Regressor predicts numerical performance scores
- **Validation Monitoring**: Train/validation/test split with performance tracking
- **Structured Output**: Combined category and score predictions

### Student Categories
The system classifies students into 10 categories:

**High-Performing Categories:**
- **High Performer**: Excellent balance (8+ hours study, 85+ previous scores, 7+ practice papers)
- **Balanced Achiever**: Good habits (7+ hours study, 7+ hours sleep, 5+ practice papers)
- **Natural Talent**: High scores (90+) with minimal effort (4 hours or less study)
- **Hard Worker**: High effort (8+ hours study) despite lower previous scores (50 or less)
- **Well Prepared**: Good balance (8+ hours sleep, 6+ hours study, 4+ practice papers)
- **Average Student**: Typical student profile with moderate engagement

**At-Risk Categories:**
- **Burnout Risk**: Excessive study (9+ hours) with insufficient sleep (4 hours or less)
- **Underachiever**: Excessive sleep (9+ hours) with minimal study (2 hours or less)
- **Struggling**: Low study (2 hours or less) and insufficient sleep (4 hours or less)
- **Needs Support**: Low engagement overall (3 hours or less study, 50 or less previous scores)

### Web Interface
- **Home Page**: Overview with all 10 categories displayed
- **Prediction Page**: Interactive form with category and score results
- **Training Page**: Model training interface with progress tracking
- **Scores Page**: Model performance metrics and validation results

### API Endpoints
- REST API for programmatic access
- JSON responses with category and performance data
- AJAX endpoints for web interface

## Technology Stack

- **Backend**: Django 6.0+
- **API Framework**: Django REST Framework
- **Machine Learning**: 
  - Scikit-learn (Random Forest Classifier & Regressor)
  - Gradient Boosting Regressor
- **Data Processing**: Pandas, NumPy
- **Database**: SQLite
- **Model Persistence**: Joblib
- **Frontend**: HTML, CSS (Neobrutalism design), JavaScript

## Dataset

The system uses a student performance dataset with the following features:

- Hours Studied
- Previous Scores
- Extracurricular Activities (Yes/No)
- Sleep Hours
- Sample Question Papers Practiced
- Performance Index (target variable)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd student_performance_index
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install django djangorestframework scikit-learn pandas numpy joblib
   ```

4. **Navigate to project directory**:
   ```bash
   cd student_ml
   ```

5. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

## Quick Start

### Option 1: Using Test Script (Recommended)

Train the model and verify everything works:

```bash
python test_new_model.py
```

This will:
- Train both classifier and regressor models
- Display detailed metrics and validation scores
- Show sample predictions with categories
- Save the model package to `performance/model.pkl`

### Option 2: Manual Setup

1. **Load Data into Database** (optional):
   ```bash
   python manage.py shell -c "from performance.load_data import run; run()"
   ```

2. **Train the Models**:
   ```bash
   python manage.py shell -c "from performance.train_model import train; train()"
   ```

3. **Start the Server**:
   ```bash
   python manage.py runserver
   ```

4. **Access the Application**:
   - Home: http://localhost:8000/
   - Predict: http://localhost:8000/predict-page/
   - Train: http://localhost:8000/train-page/
   - Scores: http://localhost:8000/scores-page/

## API Documentation

### Predict Student Performance

**Endpoint**: `POST /predict-ajax/`

**Request Body**:
```json
{
  "hours_studied": 7,
  "previous_scores": 85,
  "extracurricular": "true",
  "sleep_hours": 8,
  "sample_papers": 5
}
```

**Response**:
```json
{
  "success": true,
  "student_category": "balanced_achiever",
  "predicted_performance_index": 78.45,
  "category_description": "Good study habits with adequate sleep and preparation"
}
```

**Alternative Endpoint**: `POST /api/predict/` (REST API format)

**Field Descriptions**:
- `hours_studied`: Float (0-168) - Number of hours studied per week
- `previous_scores`: Integer (0-100) - Previous academic scores
- `extracurricular`: Boolean/String - Whether student participates in extracurricular activities
- `sleep_hours`: Float (0-24) - Hours of sleep per day
- `sample_papers`: Integer (0-100) - Number of sample question papers practiced

**Response Fields**:
- `success`: Boolean - Whether prediction was successful
- `student_category`: String - One of 10 category classifications
- `predicted_performance_index`: Float - Predicted score (0-100)
- `category_description`: String - Human-readable category explanation

### Train Model

**Endpoint**: `POST /train-ajax/`

Triggers model retraining with current dataset.

### Get Model Scores

**Endpoint**: `GET /scores-ajax/`

Returns current model performance metrics including R2 score, RMSE, MAE, and feature importances.

## Project Structure

```
student_performance_index/
├── README.md                      # This file
├── UPDATED_MODEL_README.md        # Detailed model documentation
├── QUICK_START.md                 # Quick start guide
├── TROUBLESHOOTING_CATEGORY.md    # Troubleshooting guide
├── FIX_PREDICTION_ERROR.md        # Error fixing guide
├── HOME_PAGE_UPDATE.md            # Home page documentation
├── PREDICTION_PAGE_UPDATE.md      # Prediction page documentation
└── student_ml/                    # Django project
    ├── dataset.csv                # Student performance dataset (10,000 records)
    ├── db.sqlite3                 # SQLite database
    ├── manage.py                  # Django management script
    ├── test_new_model.py          # Model training and testing script
    ├── test_prediction_fix.py     # Prediction verification script
    ├── test_endpoint.py           # API endpoint testing script
    ├── performance/               # Main Django app
    │   ├── models.py              # StudentPerformance model
    │   ├── views.py               # API views and AJAX handlers
    │   ├── urls.py                # App URL patterns
    │   ├── serializers.py         # Data serializers
    │   ├── train_model.py         # ML model training with categorization
    │   ├── load_data.py           # Data loading script
    │   ├── model.pkl              # Trained model package (classifier + regressor)
    │   ├── templates/             # HTML templates
    │   │   └── performance/
    │   │       ├── base.html      # Base template
    │   │       ├── home.html      # Home page with categories
    │   │       ├── predict.html   # Prediction interface
    │   │       ├── train.html     # Training interface
    │   │       └── scores.html    # Metrics display
    │   ├── migrations/            # Database migrations
    │   └── tests.py               # Unit tests
    └── student_ml/                # Project settings
        ├── settings.py            # Django settings
        ├── urls.py                # Main URL patterns
        ├── wsgi.py                # WSGI configuration
        └── asgi.py                # ASGI configuration
```

## Model Details

### Architecture
The system uses a dual-model approach:

**Classification Model:**
- Algorithm: Random Forest Classifier
- Purpose: Categorize students into 10 behavioral patterns
- Parameters: 200 estimators, max_depth=15, min_samples_split=5
- Output: Student category (e.g., "high_performer", "burnout_risk")

**Regression Model:**
- Algorithms: Random Forest Regressor or Gradient Boosting Regressor
- Purpose: Predict numerical performance index (0-100)
- Selection: Best model chosen based on validation R2 score
- Parameters: 
  - Random Forest: 300 estimators, max_depth=15
  - Gradient Boosting: 200 estimators, learning_rate=0.1

### Training Pipeline
1. Load dataset (10,000 student records)
2. Create student categories using rule-based classification
3. Encode categorical variables
4. Split data: Train (64%), Validation (16%), Test (20%)
5. Train classifier with stratified sampling
6. Validate classifier performance
7. Train multiple regression models
8. Select best regressor based on validation metrics
9. Package and save both models with encoders

### Performance Metrics

**Classification:**
- Validation Accuracy: ~85-95%
- Test Accuracy: ~85-95%
- Per-category precision, recall, F1-score

**Regression:**
- Validation R2: ~0.95-0.99
- Test R2: ~0.95-0.99
- RMSE: ~2-5
- MAE: ~1-3
- 5-fold Cross-validation scores

### Model Package Structure
The saved model (`model.pkl`) contains:
```python
{
    'classifier': RandomForestClassifier,
    'regressor': RandomForestRegressor or GradientBoostingRegressor,
    'label_encoder_category': LabelEncoder for categories,
    'label_encoder_activities': LabelEncoder for extracurricular activities
}
```

## Usage Examples

### Python API
```python
import joblib
import numpy as np

# Load model package
model_package = joblib.load('performance/model.pkl')

# Prepare features
features = np.array([[8, 85, 1, 8, 5]])  
# [hours_studied, previous_scores, extracurricular(0/1), sleep_hours, sample_papers]

# Get predictions
classifier = model_package['classifier']
regressor = model_package['regressor']
label_encoder = model_package['label_encoder_category']

category_encoded = classifier.predict(features)[0]
category_name = label_encoder.inverse_transform([int(category_encoded)])[0]
performance_score = regressor.predict(features)[0]

print(f"Category: {category_name}")
print(f"Performance Score: {performance_score:.2f}")
```

### cURL API
```bash
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

### Web Interface
1. Navigate to http://localhost:8000/
2. Click "Go to Predict"
3. Fill in the student information form
4. Click "Predict Performance"
5. View both category and performance score

## Testing

### Run All Tests
```bash
# Test model training
python test_new_model.py

# Test predictions
python test_prediction_fix.py

# Test API endpoints (requires server running)
python test_endpoint.py
```

### Manual Testing
```bash
# Start server
python manage.py runserver

# In another terminal, test prediction
curl -X POST http://localhost:8000/predict-ajax/ \
  -H "Content-Type: application/json" \
  -d '{"hours_studied": 7, "previous_scores": 85, "extracurricular": "true", "sleep_hours": 8, "sample_papers": 5}'
```

## Troubleshooting

### Model Not Found Error
```bash
# Train the model
python test_new_model.py
```

### Category Not Showing
```bash
# Delete old model and retrain
rm performance/model.pkl
python test_new_model.py
# Restart server
python manage.py runserver
```

### "Unexpected token" Error
- Ensure server is running
- Check if model is trained
- Verify URL endpoints in `performance/urls.py`
- See `FIX_PREDICTION_ERROR.md` for detailed troubleshooting

For more troubleshooting help, see:
- `TROUBLESHOOTING_CATEGORY.md`
- `FIX_PREDICTION_ERROR.md`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

- **UPDATED_MODEL_README.md**: Comprehensive model documentation
- **QUICK_START.md**: Quick start guide with examples
- **TROUBLESHOOTING_CATEGORY.md**: Category display troubleshooting
- **FIX_PREDICTION_ERROR.md**: API error troubleshooting
- **HOME_PAGE_UPDATE.md**: Home page features documentation
- **PREDICTION_PAGE_UPDATE.md**: Prediction interface documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Student Performance Index dataset (10,000 records)
- Machine Learning: Scikit-learn library
- Web Framework: Django and Django REST Framework
- Design: Neobrutalism UI design principles

## Future Enhancements

### Model Improvements
- Add more ML algorithms (SVM, Neural Networks, XGBoost)
- Implement ensemble methods for improved accuracy
- Add confidence scores for predictions
- Implement online learning for model updates

### Features
- User authentication and personalized tracking
- Historical performance tracking over time
- Recommendation engine for study improvements
- Data visualization dashboard with charts
- Export predictions to PDF/CSV
- Batch prediction for multiple students
- A/B testing for model versions

### Deployment
- Docker containerization
- Cloud deployment (AWS, Azure, GCP)
- CI/CD pipeline setup
- Production database (PostgreSQL)
- API rate limiting and caching
- Monitoring and logging

### Analytics
- Category distribution analysis
- Feature correlation visualization
- Model explainability (SHAP values)
- Performance trends over time
- Comparative analysis tools

## Version History

### Version 2.0 (Current)
- Added dual-model architecture (classification + regression)
- Implemented 10 student categories
- Added validation monitoring during training
- Enhanced web interface with category display
- Improved API responses with structured output
- Added comprehensive documentation

### Version 1.0
- Initial release with regression-only model
- Basic API endpoints
- Simple web interface
- Model training scripts

## Contact

For questions, issues, or suggestions, please open an issue on the repository.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB disk space
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Performance

- Training time: ~30-60 seconds (10,000 records)
- Prediction time: <100ms per request
- Model size: ~5-10MB
- Dataset size: ~1MB

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Mobile)
