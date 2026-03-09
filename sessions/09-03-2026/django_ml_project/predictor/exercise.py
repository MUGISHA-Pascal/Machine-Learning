import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio
import json
import os

# Get project's BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calculate_cv(data):
    """Calculate the coefficient of variation (CV) for the estimated_income."""
    mean = np.mean(data)
    std = np.std(data)
    if mean == 0:
        return 0
    return (std / mean) * 100

def refine_clustering_model(df, features=["estimated_income", "selling_price"]):
    """Refine clustering to achieve a higher silhouette score."""
    X = df[features]
    
    # Scaling can often improve clustering performance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using k-means after scaling usually stabilizes clusters.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    return kmeans, score, labels

def generate_rwanda_map(df):
    """Generate a high-quality Mapbox choropleth for Rwanda districts."""
    # Aggregating client counts per district
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    district_counts['district'] = district_counts['district'].str.strip()

    # Absolute path to GeoJSON
    geojson_path = os.path.join(BASE_DIR, "dummy-data", "rwanda_districts.geojson")
    
    if os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            rwanda_geojson = json.load(f)
            
        # Ensure ID alignment at the root level of each feature
        for feature in rwanda_geojson['features']:
            feature['id'] = feature['properties']['shapeName'].strip()
            
        # Using choropleth_mapbox for a more 'premium' look (WOW factor)
        fig = px.choropleth_mapbox(
            district_counts,
            geojson=rwanda_geojson,
            locations='district',
            color='client_count',
            color_continuous_scale="Reds", 
            range_color=[district_counts['client_count'].min(), district_counts['client_count'].max()],
            mapbox_style="carto-positron", 
            center={"lat": -1.94, "lon": 30.06}, 
            zoom=7.2, 
            opacity=0.7,
            title="Vehicle Clients per District in Rwanda",
            labels={'client_count': 'Total Clients'}
        )
        
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            height=600,
            dragmode="zoom",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Explicitly update mapboxes configuration. 
        # Note: scrollZoom is a config option in to_html, not a layout property in Python API.
        fig.update_mapboxes(
            center={"lat": -1.94, "lon": 30.06},
            zoom=7.2
        )
        
        # ensure boundaries are sharp
        fig.update_traces(marker_line_width=1, marker_line_color="darkred")
        
        # IMPORTANT: scrollZoom is passed via config in to_html for Plotly.js
        return pio.to_html(
            fig, 
            full_html=False, 
            include_plotlyjs=False,
            config={'scrollZoom': True}
        )
    else:
        return f"<div class='alert alert-danger'>GeoJSON not found at {geojson_path}.</div>"
