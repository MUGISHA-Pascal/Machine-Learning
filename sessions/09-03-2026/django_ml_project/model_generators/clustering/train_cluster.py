from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import QuantileTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "clustering_model.pkl"

CLUSTER_CONFIGS = [
    {
        "features": ["selling_price"],
        "scaler_name": "quantile",
        "scaler": QuantileTransformer(output_distribution="normal", random_state=42),
        "k_values": [2, 3, 4, 5],
    },
    {
        "features": ["estimated_income", "selling_price"],
        "scaler_name": "quantile",
        "scaler": QuantileTransformer(output_distribution="normal", random_state=42),
        "k_values": [2, 3, 4, 5],
    },
    {
        "features": ["estimated_income", "selling_price"],
        "scaler_name": "none",
        "scaler": None,
        "k_values": [2, 3, 4, 5],
    },
]



def _fit_kmeans(X: np.ndarray, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    samples = silhouette_samples(X, labels)
    return kmeans, labels, score, samples



def train_and_save_clustering_bundle():
    """
    Train clustering model with feature/scaler/K search, compute silhouette score
    and coefficient of variation.
    """
    df = pd.read_csv(DATASET_PATH)

    best = {
        "score": -1.0,
        "cv": None,
        "kmeans": None,
        "labels": None,
        "features": None,
        "samples": None,
        "scaler": None,
        "scaler_name": "none",
        "k": None,
    }

    for config in CLUSTER_CONFIGS:
        feats = config["features"]
        if not set(feats).issubset(df.columns):
            continue
        X_raw = df[feats].astype(float)
        if X_raw.nunique().min() < 2:
            continue

        X_values = X_raw.values
        fitted_scaler = None
        if config["scaler"] is not None:
            fitted_scaler = config["scaler"]
            X_values = fitted_scaler.fit_transform(X_values)

        for k in config["k_values"]:
            if k >= len(X_values):
                continue
            kmeans, labels, score, samples = _fit_kmeans(X_values, k)
            if score > best["score"]:
                mean_samples = float(np.mean(samples))
                best.update(
                    {
                        "score": score,
                        "cv": float(np.std(samples) / mean_samples)
                        if mean_samples != 0
                        else 0.0,
                        "kmeans": kmeans,
                        "labels": labels,
                        "features": feats,
                        "samples": samples,
                        "scaler": fitted_scaler,
                        "scaler_name": config["scaler_name"],
                        "k": k,
                    }
                )

    if best["kmeans"] is None:
        fallback_feats = ["estimated_income", "selling_price"]
        X_raw = df[fallback_feats].astype(float).values
        kmeans, labels, score, samples = _fit_kmeans(X_raw, 2)
        mean_samples = float(np.mean(samples))
        best.update(
            {
                "score": score,
                "cv": float(np.std(samples) / mean_samples) if mean_samples != 0 else 0.0,
                "kmeans": kmeans,
                "labels": labels,
                "features": fallback_feats,
                "samples": samples,
                "scaler": None,
                "scaler_name": "none",
                "k": 2,
            }
        )

    df["cluster_id"] = best["labels"]

    centers = best["kmeans"].cluster_centers_
    sorted_clusters = centers[:, 0].argsort()
    tier_names = ["Economy", "Standard", "Premium", "Executive", "Elite", "Luxury"]
    cluster_mapping = {
        int(cluster_id): (
            tier_names[idx] if idx < len(tier_names) else f"Tier {idx + 1}"
        )
        for idx, cluster_id in enumerate(sorted_clusters)
    }
    df["client_class"] = df["cluster_id"].map(cluster_mapping)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best["kmeans"],
        "mapping": cluster_mapping,
        "features": best["features"],
        "scaler": best["scaler"],
        "scaler_name": best["scaler_name"],
        "k": best["k"],
    }
    joblib.dump(bundle, MODEL_PATH)

    silhouette_avg = round(best["score"], 2)
    cv = round(best["cv"], 2) if best["cv"] is not None else 0.0

    cluster_summary = (
        df.groupby("client_class")[["estimated_income", "selling_price"]]
        .mean()
        .reset_index()
    )
    cluster_counts = df["client_class"].value_counts().reset_index()
    cluster_counts.columns = ["client_class", "count"]
    cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

    comparison_cols = [
        c
        for c in ["client_name", "estimated_income", "selling_price", "client_class"]
        if c in df.columns
    ]
    if not comparison_cols:
        comparison_cols = ["estimated_income", "selling_price", "client_class"]
    comparison_df = df[comparison_cols]

    return bundle, silhouette_avg, cv, cluster_summary, comparison_df



def get_clustering_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    bundle, _, _, _, _ = train_and_save_clustering_bundle()
    return bundle



def predict_cluster_id(bundle: dict, estimated_income: float, selling_price: float) -> int:
    feature_values = {
        "estimated_income": float(estimated_income),
        "selling_price": float(selling_price),
    }
    feats = bundle.get("features", ["estimated_income", "selling_price"])
    X = np.array([[feature_values[f] for f in feats]], dtype=float)

    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)

    return int(bundle["model"].predict(X)[0])



def evaluate_clustering_model():
    _, silhouette_avg, cv, cluster_summary, comparison_df = train_and_save_clustering_bundle()
    return {
        "silhouette": silhouette_avg,
        "cv": cv,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }