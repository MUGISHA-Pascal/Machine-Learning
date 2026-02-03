from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import numpy as np
import os
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .train_model import train as train_model_func
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create your views here.
model_package = None
training_history = []

def load_model():
    global model_package
    if model_package is None:
        model_path = 'performance/model.pkl'
        if os.path.exists(model_path):
            model_package = joblib.load(model_path)
        else:
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using: "
                "python manage.py shell -c \"from performance.train_model import train; train()\""
            )
    return model_package

def get_category_description(category):
    """Provide human-readable descriptions for categories"""
    descriptions = {
        'burnout_risk': 'High study hours with insufficient sleep - risk of burnout',
        'underachiever': 'Excessive sleep with minimal study time',
        'struggling': 'Low study hours and insufficient sleep',
        'high_performer': 'Excellent balance of study, preparation, and past performance',
        'balanced_achiever': 'Good study habits with adequate sleep and preparation',
        'natural_talent': 'High scores with minimal study effort',
        'hard_worker': 'High effort despite lower previous scores',
        'well_prepared': 'Good balance of sleep, study, and practice',
        'needs_support': 'Low engagement across multiple factors',
        'average_student': 'Typical student profile'
    }
    return descriptions.get(category, 'Standard student profile')

def home_view(request):
    """Home page with navigation"""
    return render(request, 'performance/home.html')

def predict_view(request):
    """View for prediction interface"""
    return render(request, 'performance/predict.html')

def train_view(request):
    """View for training interface"""
    return render(request, 'performance/train.html')

def scores_view(request):
    """View for displaying training scores"""
    model_exists = os.path.exists('performance/model.pkl')
    dataset_exists = os.path.exists('dataset.csv')
    
    context = {
        'model_exists': model_exists,
        'dataset_exists': dataset_exists,
        'training_history': training_history
    }
    return render(request, 'performance/scores.html', context)

@api_view(['POST'])
def predict_performance(request):
    try:
        model_package = load_model()
    except FileNotFoundError as e:
        return Response({'error': str(e)}, status=500)
    
    data = request.data

    features = np.array([[
        float(data['hours_studied']),
        float(data['previous_scores']),
        1 if data['extracurricular'] else 0,
        float(data['sleep_hours']),
        float(data['sample_papers'])
    ]])

    # Get predictions from both models
    classifier = model_package['classifier']
    regressor = model_package['regressor']
    label_encoder = model_package['label_encoder_category']
    
    # Predict category
    category_encoded = classifier.predict(features)[0]
    category_name = label_encoder.inverse_transform([int(category_encoded)])[0]
    
    # Predict performance score
    performance_score = regressor.predict(features)[0]

    return Response({
        'student_category': category_name,
        'predicted_performance_index': round(float(performance_score), 2),
        'category_description': get_category_description(category_name)
    })

@csrf_exempt
def predict_ajax(request):
    """AJAX endpoint for predictions from web interface"""
    if request.method == 'POST':
        try:
            model_package = load_model()
            data = json.loads(request.body)
            
            features = np.array([[
                float(data['hours_studied']),
                float(data['previous_scores']),
                1 if data['extracurricular'] == 'true' or data['extracurricular'] == True else 0,
                float(data['sleep_hours']),
                float(data['sample_papers'])
            ]])
            
            # Get predictions from both models
            classifier = model_package['classifier']
            regressor = model_package['regressor']
            label_encoder = model_package['label_encoder_category']
            
            # Predict category
            category_encoded = classifier.predict(features)[0]
            category_name = label_encoder.inverse_transform([int(category_encoded)])[0]
            
            # Predict performance score
            performance_score = regressor.predict(features)[0]
            
            return JsonResponse({
                'success': True,
                'student_category': category_name,
                'predicted_performance_index': round(float(performance_score), 2),
                'category_description': get_category_description(category_name)
            })
        except FileNotFoundError as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

@csrf_exempt
def train_ajax(request):
    """AJAX endpoint for training from web interface"""
    if request.method == 'POST':
        try:
            global model_package, training_history
            
            # Import here to capture output
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capture training output
            output = io.StringIO()
            with redirect_stdout(output):
                train_model_func()
            
            training_output = output.getvalue()
            
            # Reload the model
            model_package = None
            current_model_package = load_model()
            
            # Calculate metrics on dataset
            df = pd.read_csv('dataset.csv')
            from sklearn.preprocessing import LabelEncoder
            df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
            
            X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
            y = df['Performance Index']
            
            regressor = current_model_package['regressor']
            predictions = regressor.predict(X)
            
            metrics = {
                'r2_score': float(r2_score(y, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                'mae': float(mean_absolute_error(y, predictions)),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add to history
            training_history.append(metrics)
            
            return JsonResponse({
                'success': True,
                'message': 'Model trained successfully with classification and regression!',
                'metrics': metrics,
                'output': training_output
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

@csrf_exempt
def get_scores_ajax(request):
    """AJAX endpoint to get current model scores"""
    if request.method == 'GET':
        try:
            if not os.path.exists('performance/model.pkl'):
                return JsonResponse({
                    'success': False,
                    'error': 'Model not found. Please train the model first.'
                })
            
            model_package = load_model()
            regressor = model_package['regressor']
            classifier = model_package['classifier']
            
            # Calculate metrics on dataset
            df = pd.read_csv('dataset.csv')
            from sklearn.preprocessing import LabelEncoder
            df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
            
            X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
            y = df['Performance Index']
            
            predictions = regressor.predict(X)
            
            metrics = {
                'r2_score': float(r2_score(y, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                'mae': float(mean_absolute_error(y, predictions)),
                'dataset_size': len(df)
            }
            
            # Feature importance if available
            if hasattr(regressor, 'feature_importances_'):
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(X.columns, regressor.feature_importances_)
                }
                metrics['feature_importance'] = feature_importance
            
            return JsonResponse({
                'success': True,
                'metrics': metrics,
                'training_history': training_history,
                'model_type': 'Combined Classification + Regression'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)