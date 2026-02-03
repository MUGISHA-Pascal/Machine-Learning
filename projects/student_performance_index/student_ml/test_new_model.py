#!/usr/bin/env python
"""
Test script for the updated student performance model
Tests both classification and regression functionality
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
django.setup()

from performance.train_model import train
import joblib
import numpy as np

def test_model():
    print("="*70)
    print("TESTING UPDATED STUDENT PERFORMANCE MODEL")
    print("="*70)
    
    # Train the model
    print("\n1. Training model with classification and regression...")
    print("-"*70)
    train()
    
    # Load the trained model
    print("\n2. Loading trained model package...")
    print("-"*70)
    model_package = joblib.load('performance/model.pkl')
    
    print("✓ Model package loaded successfully!")
    print(f"  - Classifier: {type(model_package['classifier']).__name__}")
    print(f"  - Regressor: {type(model_package['regressor']).__name__}")
    print(f"  - Categories: {list(model_package['label_encoder_category'].classes_)}")
    
    # Test predictions
    print("\n3. Testing predictions...")
    print("-"*70)
    
    test_cases = [
        {
            'name': 'Balanced High Performer',
            'features': [8, 90, 1, 8, 7],
            'description': '8h study, 90 prev score, activities, 8h sleep, 7 papers'
        },
        {
            'name': 'Burnout Risk',
            'features': [9, 70, 0, 4, 3],
            'description': '9h study, 70 prev score, no activities, 4h sleep, 3 papers'
        },
        {
            'name': 'Natural Talent',
            'features': [2, 95, 1, 9, 1],
            'description': '2h study, 95 prev score, activities, 9h sleep, 1 paper'
        },
        {
            'name': 'Struggling Student',
            'features': [2, 40, 0, 4, 1],
            'description': '2h study, 40 prev score, no activities, 4h sleep, 1 paper'
        },
        {
            'name': 'Hard Worker',
            'features': [9, 45, 1, 7, 8],
            'description': '9h study, 45 prev score, activities, 7h sleep, 8 papers'
        }
    ]
    
    classifier = model_package['classifier']
    regressor = model_package['regressor']
    label_encoder = model_package['label_encoder_category']
    
    for test_case in test_cases:
        features = np.array([test_case['features']])
        
        # Get predictions
        category_encoded = classifier.predict(features)[0]
        category_name = label_encoder.inverse_transform([int(category_encoded)])[0]
        performance_score = regressor.predict(features)[0]
        
        print(f"\n{test_case['name']}:")
        print(f"  Input: {test_case['description']}")
        print(f"  → Category: {category_name}")
        print(f"  → Performance Score: {performance_score:.2f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nThe model now provides:")
    print("  1. Student category classification (10 categories)")
    print("  2. Performance index prediction (regression)")
    print("  3. Validation monitoring during training")
    print("\nYou can now use the API endpoints to get predictions with both")
    print("category and performance score!")

if __name__ == '__main__':
    test_model()
