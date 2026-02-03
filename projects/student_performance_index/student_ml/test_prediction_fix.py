#!/usr/bin/env python
"""
Quick test to verify prediction fix works
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
django.setup()

import joblib
import numpy as np

def test_predictions():
    print("="*70)
    print("TESTING PREDICTION FIX")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('performance/model.pkl'):
        print("\n❌ Model not found. Please train the model first:")
        print("   python test_new_model.py")
        return
    
    # Load model
    print("\n1. Loading model package...")
    model_package = joblib.load('performance/model.pkl')
    print("✓ Model loaded successfully")
    
    # Extract components
    classifier = model_package['classifier']
    regressor = model_package['regressor']
    label_encoder = model_package['label_encoder_category']
    
    print(f"\n2. Model components:")
    print(f"   - Classifier: {type(classifier).__name__}")
    print(f"   - Regressor: {type(regressor).__name__}")
    print(f"   - Categories: {len(label_encoder.classes_)} categories")
    
    # Test predictions
    print("\n3. Testing predictions...")
    print("-"*70)
    
    test_cases = [
        {
            'name': 'Test Case 1: Balanced Student',
            'features': [7, 85, 1, 8, 5],
            'description': '7h study, 85 score, activities, 8h sleep, 5 papers'
        },
        {
            'name': 'Test Case 2: Burnout Risk',
            'features': [9, 70, 0, 4, 2],
            'description': '9h study, 70 score, no activities, 4h sleep, 2 papers'
        },
        {
            'name': 'Test Case 3: Natural Talent',
            'features': [2, 95, 1, 9, 1],
            'description': '2h study, 95 score, activities, 9h sleep, 1 paper'
        }
    ]
    
    for test_case in test_cases:
        try:
            # Prepare features as 2D array
            features = np.array([test_case['features']], dtype=float)
            
            # Get predictions
            category_encoded = classifier.predict(features)[0]
            
            # Convert to int explicitly to avoid indexing issues
            category_encoded_int = int(category_encoded)
            
            # Decode category
            category_name = label_encoder.inverse_transform([category_encoded_int])[0]
            
            # Get performance score
            performance_score = regressor.predict(features)[0]
            
            print(f"\n✓ {test_case['name']}")
            print(f"  Input: {test_case['description']}")
            print(f"  Features: {test_case['features']}")
            print(f"  → Category: {category_name}")
            print(f"  → Performance Score: {performance_score:.2f}")
            
        except Exception as e:
            print(f"\n❌ {test_case['name']} FAILED")
            print(f"  Error: {str(e)}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*70)
    print("✓ ALL PREDICTIONS SUCCESSFUL!")
    print("="*70)
    print("\nThe indexing error has been fixed. You can now:")
    print("  1. Use the API endpoints")
    print("  2. Make predictions via Python")
    print("  3. Use the web interface")
    return True

if __name__ == '__main__':
    success = test_predictions()
    sys.exit(0 if success else 1)
