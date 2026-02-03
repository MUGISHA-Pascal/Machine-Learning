#!/usr/bin/env python
"""
Test script to verify the predict-ajax endpoint is working
"""

import requests
import json

def test_endpoint():
    print("="*70)
    print("TESTING PREDICT-AJAX ENDPOINT")
    print("="*70)
    
    # Test data
    test_data = {
        'hours_studied': 8,
        'previous_scores': 85,
        'extracurricular': 'true',
        'sleep_hours': 8,
        'sample_papers': 5
    }
    
    # Test different URL patterns
    urls = [
        'http://localhost:8000/predict-ajax/',
        'http://localhost:8000/api/predict/',
        'http://127.0.0.1:8000/predict-ajax/',
        'http://127.0.0.1:8000/api/predict/',
    ]
    
    print("\nTesting endpoints...")
    print("-"*70)
    
    for url in urls:
        print(f"\nTrying: {url}")
        try:
            response = requests.post(
                url,
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            print(f"  Status Code: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'Not set')}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  ✓ SUCCESS - JSON Response:")
                    print(f"    - success: {data.get('success')}")
                    print(f"    - student_category: {data.get('student_category', 'N/A')}")
                    print(f"    - predicted_performance_index: {data.get('predicted_performance_index', 'N/A')}")
                    print(f"    - category_description: {data.get('category_description', 'N/A')[:50]}...")
                    return True
                except json.JSONDecodeError:
                    print(f"  ✗ FAILED - Response is not JSON")
                    print(f"  Response preview: {response.text[:200]}")
            elif response.status_code == 404:
                print(f"  ✗ FAILED - Endpoint not found (404)")
            elif response.status_code == 500:
                print(f"  ✗ FAILED - Server error (500)")
                print(f"  Response preview: {response.text[:200]}")
            else:
                print(f"  ✗ FAILED - Unexpected status code")
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ FAILED - Cannot connect. Is the server running?")
        except requests.exceptions.Timeout:
            print(f"  ✗ FAILED - Request timeout")
        except Exception as e:
            print(f"  ✗ FAILED - Error: {str(e)}")
    
    print("\n" + "="*70)
    print("TROUBLESHOOTING STEPS:")
    print("="*70)
    print("1. Make sure Django server is running:")
    print("   python manage.py runserver")
    print("\n2. Check if model is trained:")
    print("   python test_new_model.py")
    print("\n3. Check Django logs in the terminal where server is running")
    print("\n4. Verify URLs are configured in performance/urls.py")
    print("="*70)
    
    return False

if __name__ == '__main__':
    import sys
    
    print("\nMake sure Django server is running before testing!")
    print("In another terminal, run: python manage.py runserver\n")
    
    input("Press Enter when server is ready...")
    
    success = test_endpoint()
    sys.exit(0 if success else 1)
