from django.shortcuts import render
import joblib
import numpy as np
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
model = None

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