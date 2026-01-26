# Student Performance Prediction System

A machine learning-powered Django REST API that predicts student performance index based on various academic and lifestyle factors.

## Overview

This project implements a predictive model to estimate student performance using machine learning. The system analyzes factors such as study hours, previous scores, extracurricular activities, sleep patterns, and practice question papers to predict a performance index.

## Features

- **Machine Learning Model**: Random Forest Regressor for accurate performance prediction
- **REST API**: Django REST Framework-based API for easy integration
- **Data Management**: Automated data loading from CSV to Django models
- **Model Training**: Script to train and save ML models
- **Database Integration**: SQLite database with Django ORM

## Technology Stack

- **Backend**: Django 6.0+
- **API Framework**: Django REST Framework
- **Machine Learning**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Database**: SQLite
- **Serialization**: Joblib for model persistence

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
   cd "Student Performance Index"
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
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

## Usage

### 1. Load Data into Database

Load the student performance data into the Django database:

```bash
python manage.py shell -c "from performance.load_data import run; run()"
```

### 2. Train the Machine Learning Model

Train the Random Forest model:

```bash
python manage.py shell -c "from performance.train_model import train; train()"
```

This will create `performance/model.pkl` file containing the trained model.

### 3. Run the Development Server

Start the Django development server:

```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`

## API Documentation

### Predict Student Performance

**Endpoint**: `POST /api/predict/`

**Request Body**:

```json
{
  "hours_studied": 7,
  "previous_scores": 85,
  "extracurricular": true,
  "sleep_hours": 8,
  "sample_papers": 3
}
```

**Response**:

```json
{
  "Predicted_performance_index": 78.45
}
```

**Field Descriptions**:

- `hours_studied`: Integer (1-10) - Number of hours studied per day
- `previous_scores`: Integer (0-100) - Previous academic scores
- `extracurricular`: Boolean - Whether student participates in extracurricular activities
- `sleep_hours`: Integer (1-12) - Hours of sleep per day
- `sample_papers`: Integer (0-10) - Number of sample question papers practiced

## Project Structure

```
student_ml/
├── dataset.csv                    # Student performance dataset
├── db.sqlite3                     # SQLite database
├── manage.py                      # Django management script
├── performance/                   # Main Django app
│   ├── models.py                  # StudentPerformance model
│   ├── views.py                   # API views
│   ├── urls.py                    # App URL patterns
│   ├── serializers.py             # Data serializers
│   ├── train_model.py             # ML model training script
│   ├── load_data.py               # Data loading script
│   ├── migrations/                # Database migrations
│   └── tests.py                   # Unit tests
└── student_ml/                    # Project settings
    ├── settings.py                # Django settings
    ├── urls.py                    # Main URL patterns
    ├── wsgi.py                    # WSGI configuration
    └── asgi.py                    # ASGI configuration
```

## Model Details

The system uses a Random Forest Regressor with:

- **n_estimators**: 200
- **random_state**: 42 for reproducibility

The model is trained on 80% of the data and validated on 20%.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- Add more ML algorithms (SVM, Neural Networks)
- Implement model comparison and selection
- Add data visualization dashboard
- Deploy to cloud platform
- Add authentication and user management
- Implement model retraining pipeline
