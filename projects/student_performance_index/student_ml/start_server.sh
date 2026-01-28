#!/bin/bash

echo "🎓 Student Performance Predictor - Starting Server"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Please create one first: python -m venv ../venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ../venv/bin/activate

# Check if model exists
if [ ! -f "performance/model.pkl" ]; then
    echo ""
    echo "⚠️  Model not found!"
    echo "The model will need to be trained through the web interface."
    echo "After starting the server, go to: http://localhost:8000/train-page/"
    echo ""
fi

# Start Django server
echo ""
echo "🚀 Starting Django development server..."
echo "📍 Access the application at: http://localhost:8000/"
echo ""
echo "Available pages:"
echo "  - Home:         http://localhost:8000/"
echo "  - Predict:      http://localhost:8000/predict-page/"
echo "  - Train Model:  http://localhost:8000/train-page/"
echo "  - View Scores:  http://localhost:8000/scores-page/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python manage.py runserver
