from django.shortcuts import render
import joblib
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
model = joblib.load('performance/model.pkl')

@api_view(['POST'])
def predict_performance(request):
    data = request.data

    features = np.array([[
        data['hours_studied'],
        data['previous_scores'],
        1 if data['extracurricular'] else 0,
        data['sleep_hours'],
        data['sample_papers']
    ]])

    prediction = model.predict(features)[0]

    return Response({
        'Predicted_performance_index' : round(float(prediction), 2)
    })