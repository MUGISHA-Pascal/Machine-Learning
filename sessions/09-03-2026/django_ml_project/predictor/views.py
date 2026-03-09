import pandas as pd
from django.shortcuts import render
import joblib
import os
import numpy as np

# Data Exploration & Plotting
from predictor.data_exploration import dataset_exploration, data_exploration
from predictor.exercise import calculate_cv, refine_clustering_model, generate_rwanda_map

# Evaluations
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Get base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(path):
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        return joblib.load(full_path)
    return None

# Load initial models
regression_model = load_model("model_generators/regression/regression_model.pkl")
classification_model = load_model("model_generators/classification/classification_model.pkl")
clustering_model = load_model("model_generators/clustering/clustering_model.pkl")

def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Task (a): Rwanda Map
    rwanda_map_html = generate_rwanda_map(df)
    
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_map_html,
    }
    return render(request, "predictor/index.html", context)

def regression_analysis(request):
    eval_results = evaluate_regression_model()
    context = {
        "evaluations": eval_results
    }
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            if regression_model:
                prediction = regression_model.predict([[year, km, seats, income]])[0]
                context["price"] = prediction
            else:
                context["error"] = "Regression model not loaded"
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/regression_analysis.html", context)

def classification_analysis(request):
    eval_results = evaluate_classification_model()
    context = {
        "evaluations": eval_results
    }
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            if classification_model:
                prediction = classification_model.predict([[year, km, seats, income]])[0]
                context["prediction"] = prediction
            else:
                context["error"] = "Classification model not loaded"
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/classification_analysis.html", context)

def clustering_analysis(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Calculate Coefficient of Variation for exercise
    cv_score = calculate_cv(df["estimated_income"])
    
    # Refine Clustering model to get S.S. > 0.9 for exercise
    _, refined_ss, _ = refine_clustering_model(df)
    
    eval_results = evaluate_clustering_model()
    # Add CV and Refined Score to evaluations
    eval_results["cv_score"] = round(cv_score, 2)
    eval_results["refined_silhouette"] = round(refined_ss, 2)

    context = {
        "evaluations": eval_results
    }
    
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            
            if regression_model and clustering_model:
                predicted_price = regression_model.predict([[year, km, seats, income]])[0]
                cluster_id = clustering_model.predict([[income, predicted_price]])[0]
                
                mapping = {0: "Economy", 1: "Standard", 2: "Premium"}
                context.update({
                    "prediction": mapping.get(cluster_id, "Unknown"),
                    "price": predicted_price
                })
            else:
                context["error"] = "Required models not loaded"
        except Exception as e:
            context["error"] = str(e)
            
    return render(request, "predictor/clustering_analysis.html", context)
