import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio

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
    
    # Scaling can often improve clustering performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different cluster counts or models if needed, 
    # but scaling is usually the first step for k-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    return kmeans, score, labels

def generate_rwanda_map(df):
    """Generate a Plotly map for Rwanda districts."""
    # Group by district and count clients
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    # Rwanda districts GeoJSON URL (reliable source)
    geojson_url = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/9469f09/releaseData/gbOpen/RWA/ADM2/geoBoundaries-RWA-ADM2.geojson"
    
    fig = px.choropleth(
        district_counts,
        geojson=geojson_url,
        locations='district',
        featureidkey="properties.shapeName", # geoBoundaries uses shapeName for the main name
        color='client_count',
        color_continuous_scale="Viridis",
        title="Vehicle Clients per District in Rwanda",
        labels={'client_count': 'Number of Clients'}
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return pio.to_html(fig, full_html=False)
