from django.urls import path
from .views import (
    predict_performance, 
    home_view, 
    predict_view, 
    train_view, 
    scores_view,
    predict_ajax,
    train_ajax,
    get_scores_ajax
)

urlpatterns = [
    # Web views
    path('', home_view, name='home'),
    path('predict-page/', predict_view, name='predict_page'),
    path('train-page/', train_view, name='train_page'),
    path('scores-page/', scores_view, name='scores_page'),
    
    # API endpoints
    path('predict/', predict_performance, name='api_predict'),
    path('api/predict/', predict_ajax, name='ajax_predict'),
    path('api/train/', train_ajax, name='ajax_train'),
    path('api/scores/', get_scores_ajax, name='ajax_scores'),
]