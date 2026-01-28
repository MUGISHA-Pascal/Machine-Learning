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
model = None
training_history = []

def load_model():
    global model
    if model is None:
        model_path = 'performance/model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using: "
                "python manage.py shell -c \"from performance.train_model import train; train()\""
            )
    return model

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
        current_model = load_model()
    except FileNotFoundError as e:
        return Response({'error': str(e)}, status=500)
    
    data = request.data

    features = np.array([[
        data['hours_studied'],
        data['previous_scores'],
        1 if data['extracurricular'] else 0,
        data['sleep_hours'],
        data['sample_papers']
    ]])

    prediction = current_model.predict(features)[0]

    return Response({
        'Predicted_performance_index' : round(float(prediction), 2)
    })

@csrf_exempt
def predict_ajax(request):
    """AJAX endpoint for predictions from web interface"""
    if request.method == 'POST':
        try:
            current_model = load_model()
            data = json.loads(request.body)
            
            features = np.array([[
                float(data['hours_studied']),
                float(data['previous_scores']),
                1 if data['extracurricular'] == 'true' or data['extracurricular'] == True else 0,
                float(data['sleep_hours']),
                float(data['sample_papers'])
            ]])
            
            prediction = current_model.predict(features)[0]
            
            return JsonResponse({
                'success': True,
                'predicted_performance_index': round(float(prediction), 2)
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
            global model, training_history
            
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
            model = None
            current_model = load_model()
            
            # Calculate metrics on dataset
            df = pd.read_csv('dataset.csv')
            from sklearn.preprocessing import LabelEncoder
            df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
            
            X = df.drop('Performance Index', axis=1)
            y = df['Performance Index']
            
            predictions = current_model.predict(X)
            
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
                'message': 'Model trained successfully!',
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
            
            current_model = load_model()
            
            # Calculate metrics on dataset
            df = pd.read_csv('dataset.csv')
            from sklearn.preprocessing import LabelEncoder
            df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
            
            X = df.drop('Performance Index', axis=1)
            y = df['Performance Index']
            
            predictions = current_model.predict(X)
            
            metrics = {
                'r2_score': float(r2_score(y, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                'mae': float(mean_absolute_error(y, predictions)),
                'dataset_size': len(df)
            }
            
            # Feature importance if available
            if hasattr(current_model, 'feature_importances_'):
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(X.columns, current_model.feature_importances_)
                }
                metrics['feature_importance'] = feature_importance
            
            return JsonResponse({
                'success': True,
                'metrics': metrics,
                'training_history': training_history
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)