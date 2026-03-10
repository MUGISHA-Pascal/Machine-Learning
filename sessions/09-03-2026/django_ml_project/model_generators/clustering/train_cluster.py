from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "clustering_model.pkl"

TRIM_LOW = 0.20
TRIM_HIGH = 0.80
TARGET_K = 2
MIN_SILHOUETTE = 0.90
MAX_CV = 0.30

FEATURE_SETS = [
                ["selling_price"],
                ["estimated_income", "selling_price"],
                ["wholesale_price", "selling_price"],
                ["estimated_income", "wholesale_price", "selling_price"],
                ["year", "kilometers_driven", "seating_capacity", "estimated_income", "selling_price"],
]

SCALERS = [
                ("quantile", QuantileTransformer(output_distribution="normal", random_state=42)),
                ("robust", RobustScaler()),
                ("standard", StandardScaler()),
                ("none", None),
]

METRIC_COLUMNS = ["estimated_income", "selling_price"]



def _cv(values: pd.Series | np.ndarray) -> float:
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                                return 0.0
                mean = float(np.mean(arr))
                if mean == 0:
                                return 0.0
                return float(np.std(arr, ddof=0) / mean)



def _fit_kmeans(X: np.ndarray, n_clusters: int):
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
                labels = model.fit_predict(X)
                if len(np.unique(labels)) < 2:
                                return model, labels, -1.0, np.array([])
                sil = silhouette_score(X, labels)
                sil_samples = silhouette_samples(X, labels)
                return model, labels, float(sil), sil_samples



def _cluster_trim_mask(df_metrics: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
                mask = np.ones(len(df_metrics), dtype=bool)
                for cluster_id in np.unique(labels):
                                cluster_mask = labels == cluster_id
                                if cluster_mask.sum() < 10:
                                                return np.zeros(len(df_metrics), dtype=bool)
                                idx = np.where(cluster_mask)[0]
                                for col in METRIC_COLUMNS:
                                                series = df_metrics.iloc[idx][col].astype(float)
                                                low = series.quantile(TRIM_LOW)
                                                high = series.quantile(TRIM_HIGH)
                                                col_mask_cluster = (series >= low) & (series <= high)
                                                cluster_keep = np.zeros(len(df_metrics), dtype=bool)
                                                cluster_keep[idx] = col_mask_cluster.to_numpy()
                                                mask &= (cluster_keep | (~cluster_mask))
                return mask



def _compute_candidate_metrics(
                df: pd.DataFrame,
                X: np.ndarray,
                labels: np.ndarray,
                sil_samples: np.ndarray,
) -> dict | None:
                if sil_samples.size == 0:
                                return None

                df_metrics = df[METRIC_COLUMNS].astype(float).copy()
                trim_mask = _cluster_trim_mask(df_metrics, labels)
                if trim_mask.sum() <= TARGET_K:
                                return None

                X_trim = X[trim_mask]
                labels_trim = labels[trim_mask]
                if len(np.unique(labels_trim)) < TARGET_K:
                                return None

                silhouette = float(silhouette_score(X_trim, labels_trim))
                sil_samples_trim = silhouette_samples(X_trim, labels_trim)
                sil_mean = float(np.mean(sil_samples_trim))
                sil_cv = float(np.std(sil_samples_trim, ddof=0) / sil_mean) if sil_mean != 0 else 0.0

                df_trim = df.loc[trim_mask].copy()
                df_trim["cluster_id"] = labels_trim

                total_income_cv = _cv(df_trim["estimated_income"])
                total_selling_cv = _cv(df_trim["selling_price"])

                class_rows = []
                constraints_ok = True
                for cluster_id, group in df_trim.groupby("cluster_id", dropna=False):
                                income_cv = _cv(group["estimated_income"])
                                selling_cv = _cv(group["selling_price"])
                                class_rows.append(
                                                {
                                                                "cluster_id": int(cluster_id),
                                                                "count": int(len(group)),
                                                                "estimated_income": float(group["estimated_income"].mean()),
                                                                "estimated_income_cv": float(income_cv),
                                                                "selling_price": float(group["selling_price"].mean()),
                                                                "selling_price_cv": float(selling_cv),
                                                }
                                )
                                if (
                                                income_cv >= MAX_CV
                                                or selling_cv >= MAX_CV
                                                or income_cv > total_income_cv
                                                or selling_cv > total_selling_cv
                                ):
                                                constraints_ok = False

                if silhouette < MIN_SILHOUETTE or sil_cv >= MAX_CV:
                                constraints_ok = False

                metrics = {
                                "constraints_ok": constraints_ok,
                                "silhouette": silhouette,
                                "sil_cv": sil_cv,
                                "trim_mask": trim_mask,
                                "total_income_cv": total_income_cv,
                                "total_selling_cv": total_selling_cv,
                                "class_rows": class_rows,
                }
                return metrics



def _ordered_mapping(df_trimmed_with_labels: pd.DataFrame) -> dict[int, str]:
                ranked = (
                                df_trimmed_with_labels.groupby("cluster_id")["selling_price"].mean().sort_values().index.tolist()
                )
                names = ["Economy", "Premium"]
                return {int(cluster_id): names[i] if i < len(names) else f"Tier {i + 1}" for i, cluster_id in enumerate(ranked)}



def train_and_save_clustering_bundle():
                """
                Strict clustering search:
                - K=2
                - Per-cluster 20-80 trimming on estimated_income and selling_price
                - silhouette > 0.90
                - silhouette CV < 0.30
                - per-class income/selling CV < 0.30 and <= corresponding total CV
                """
                df = pd.read_csv(DATASET_PATH)

                required = {
                                "year",
                                "kilometers_driven",
                                "seating_capacity",
                                "estimated_income",
                                "wholesale_price",
                                "selling_price",
                }
                missing = sorted(required - set(df.columns))
                if missing:
                                raise ValueError(f"Missing required columns in dataset: {missing}")

                best_feasible = None
                best_fallback = None

                for features in FEATURE_SETS:
                                if not set(features).issubset(df.columns):
                                                continue
                                X_raw_df = df[features].astype(float)
                                if X_raw_df.nunique().min() < 2:
                                                continue
                                X_raw = X_raw_df.to_numpy()

                                for scaler_name, scaler in SCALERS:
                                                fitted_scaler = None
                                                X = X_raw
                                                if scaler is not None:
                                                                fitted_scaler = scaler
                                                                X = fitted_scaler.fit_transform(X_raw)

                                                model, labels, _, sil_samples = _fit_kmeans(X, TARGET_K)
                                                candidate_metrics = _compute_candidate_metrics(df, X, labels, sil_samples)
                                                if candidate_metrics is None:
                                                                continue

                                                candidate = {
                                                                "features": features,
                                                                "scaler_name": scaler_name,
                                                                "scaler": fitted_scaler,
                                                                "model": model,
                                                                "labels": labels,
                                                                "X": X,
                                                                "metrics": candidate_metrics,
                                                }

                                                if candidate_metrics["constraints_ok"]:
                                                                if (
                                                                                best_feasible is None
                                                                                or candidate_metrics["silhouette"] > best_feasible["metrics"]["silhouette"]
                                                                                or (
                                                                                                candidate_metrics["silhouette"] == best_feasible["metrics"]["silhouette"]
                                                                                                and candidate_metrics["sil_cv"] < best_feasible["metrics"]["sil_cv"]
                                                                                )
                                                                ):
                                                                                best_feasible = candidate

                                                if (
                                                                best_fallback is None
                                                                or candidate_metrics["silhouette"] > best_fallback["metrics"]["silhouette"]
                                                                or (
                                                                                candidate_metrics["silhouette"] == best_fallback["metrics"]["silhouette"]
                                                                                and candidate_metrics["sil_cv"] < best_fallback["metrics"]["sil_cv"]
                                                                )
                                                ):
                                                                best_fallback = candidate

                selected = best_feasible if best_feasible is not None else best_fallback
                if selected is None:
                                raise RuntimeError("No valid clustering configuration found.")

                labels = selected["labels"]
                trim_mask = selected["metrics"]["trim_mask"]
                df_trimmed = df.loc[trim_mask].copy()
                df_trimmed["cluster_id"] = labels[trim_mask]

                cluster_mapping = _ordered_mapping(df_trimmed)
                df_trimmed["client_class"] = df_trimmed["cluster_id"].map(cluster_mapping)

                model_bundle = {
                                "model": selected["model"],
                                "mapping": cluster_mapping,
                                "features": selected["features"],
                                "scaler": selected["scaler"],
                                "scaler_name": selected["scaler_name"],
                                "k": TARGET_K,
                                "trim_low": TRIM_LOW,
                                "trim_high": TRIM_HIGH,
                                "constraints_met": bool(selected["metrics"]["constraints_ok"]),
                }
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model_bundle, MODEL_PATH)

                cluster_summary = (
                                df_trimmed.groupby("client_class", dropna=False)
                                .agg(
                                                count=("client_class", "size"),
                                                estimated_income=("estimated_income", "mean"),
                                                estimated_income_cv=("estimated_income", _cv),
                                                selling_price=("selling_price", "mean"),
                                                selling_price_cv=("selling_price", _cv),
                                )
                                .reset_index()
                )
                class_order = list(cluster_mapping.values())
                cluster_summary["client_class"] = pd.Categorical(
                                cluster_summary["client_class"], categories=class_order, ordered=True
                )
                cluster_summary = cluster_summary.sort_values("client_class").reset_index(drop=True)
                cluster_summary["client_class"] = cluster_summary["client_class"].astype(str)

                comparison_cols = [
                                c
                                for c in ["client_name", "estimated_income", "selling_price", "client_class"]
                                if c in df_trimmed.columns
                ]
                if not comparison_cols:
                                comparison_cols = ["estimated_income", "selling_price", "client_class"]
                comparison_df = df_trimmed[comparison_cols].copy()

                return (
                                model_bundle,
                                round(selected["metrics"]["silhouette"], 2),
                                round(selected["metrics"]["sil_cv"], 2),
                                round(selected["metrics"]["total_income_cv"], 2),
                                round(selected["metrics"]["total_selling_cv"], 2),
                                cluster_summary,
                                comparison_df,
                )



def get_clustering_bundle():
                if MODEL_PATH.exists():
                                return joblib.load(MODEL_PATH)
                bundle, *_ = train_and_save_clustering_bundle()
                return bundle



def predict_cluster_id(
                bundle: dict,
                estimated_income: float,
                selling_price: float,
                seating_capacity: float | None = None,
) -> int:
                feature_values = {
                                "estimated_income": float(estimated_income),
                                "selling_price": float(selling_price),
                                "seating_capacity": float(seating_capacity) if seating_capacity is not None else 0.0,
                }
                feats = bundle.get("features", ["selling_price"])
                X = np.array([[feature_values[f] for f in feats]], dtype=float)

                scaler = bundle.get("scaler")
                if scaler is not None:
                                X = scaler.transform(X)

                return int(bundle["model"].predict(X)[0])



def evaluate_clustering_model():
                (
                                _,
                                silhouette_avg,
                                cv,
                                total_income_cv,
                                total_selling_price_cv,
                                cluster_summary,
                                comparison_df,
                ) = train_and_save_clustering_bundle()

                return {
                                "silhouette": silhouette_avg,
                                "cv": cv,
                                "total_income_cv": total_income_cv,
                                "total_selling_price_cv": total_selling_price_cv,
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