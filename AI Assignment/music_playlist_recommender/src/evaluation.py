from pathlib import Path

import numpy as np
import pandas as pd

from src.collaborative_filtering import evaluate_collaborative_rmse, recommend_songs_collaborative
from src.content_based_filtering import recommend_similar_songs
from src.data_loader import load_clean_data
from src.hybrid_filtering import evaluate_hybrid_rmse, recommend_songs_hybrid
from src.ml_helpers import mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVALUATION_CSV = OUTPUT_DIR / "evaluation_results.csv"


def calculate_mse(actual_values, predicted_values):
    """MSE means Mean Squared Error; lower values mean smaller prediction mistakes."""
    return round(float(mean_squared_error(actual_values, predicted_values)), 4)


def calculate_rmse(actual_values, predicted_values):
    """RMSE is the square root of MSE and is easier to understand on a rating scale."""
    mse = mean_squared_error(actual_values, predicted_values)
    return round(float(np.sqrt(mse)), 4)


def precision_at_k(recommended_ids, relevant_ids, k=10):
    """Precision@K means: out of K recommendations, how many were actually relevant?"""
    recommended_top_k = list(recommended_ids)[:k]
    if not recommended_top_k:
        return 0
    return len(set(recommended_top_k) & set(relevant_ids)) / len(recommended_top_k)


def recall_at_k(recommended_ids, relevant_ids, k=10):
    """Recall@K means: out of all relevant songs, how many did the model find?"""
    if not relevant_ids:
        return 0
    recommended_top_k = list(recommended_ids)[:k]
    return len(set(recommended_top_k) & set(relevant_ids)) / len(set(relevant_ids))


def f1_at_k(precision, recall):
    """F1@K combines precision and recall into one balanced score."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def _content_recommendations_for_user(user_id, top_k):
    songs_df, interactions_df = load_clean_data()
    liked_songs = interactions_df[(interactions_df["user_id"] == user_id) & (interactions_df["liked"] == 1)]
    if liked_songs.empty:
        return []

    seed_song_id = liked_songs.iloc[0]["song_id"]
    recommendations = recommend_similar_songs(seed_song_id, top_n=top_k)
    return recommendations["song_id"].tolist() if not recommendations.empty else []


def evaluate_precision_recall_f1(model_name, top_k=10):
    """Evaluate recommendations by checking whether they include songs each user liked."""
    _, interactions_df = load_clean_data()
    users = interactions_df["user_id"].unique()

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for user_id in users:
        relevant_ids = interactions_df[
            (interactions_df["user_id"] == user_id) & (interactions_df["liked"] == 1)
        ]["song_id"].tolist()

        if model_name == "Collaborative Filtering":
            recommendations = recommend_songs_collaborative(user_id, top_n=top_k, exclude_seen=False)
            recommended_ids = recommendations["song_id"].tolist() if not recommendations.empty else []
        elif model_name == "Content-Based Filtering":
            recommended_ids = _content_recommendations_for_user(user_id, top_k)
        else:
            recommendations = recommend_songs_hybrid(user_id, top_n=top_k, exclude_seen=False)
            recommended_ids = recommendations["song_id"].tolist() if not recommendations.empty else []

        precision = precision_at_k(recommended_ids, relevant_ids, top_k)
        recall = recall_at_k(recommended_ids, relevant_ids, top_k)
        f1 = f1_at_k(precision, recall)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return (
        round(float(np.mean(precision_scores)), 4),
        round(float(np.mean(recall_scores)), 4),
        round(float(np.mean(f1_scores)), 4),
    )


def _simple_content_rmse():
    """A small rating-style estimate for content-based filtering so all models have RMSE."""
    _, interactions_df = load_clean_data()
    predictions = []
    actuals = []

    for _, row in interactions_df.iterrows():
        user_rows = interactions_df[
            (interactions_df["user_id"] == row["user_id"]) & (interactions_df["song_id"] != row["song_id"])
        ]
        if user_rows.empty:
            predicted = interactions_df["rating"].mean()
        else:
            predicted = user_rows["rating"].mean()
        predictions.append(predicted)
        actuals.append(row["rating"])

    return calculate_rmse(actuals, predictions)


def evaluate_all_models():
    """Create and save the final model comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cf_precision, cf_recall, cf_f1 = evaluate_precision_recall_f1("Collaborative Filtering")
    cb_precision, cb_recall, cb_f1 = evaluate_precision_recall_f1("Content-Based Filtering")
    hybrid_precision, hybrid_recall, hybrid_f1 = evaluate_precision_recall_f1("Hybrid Filtering")

    cf_rmse = evaluate_collaborative_rmse()
    cb_rmse = _simple_content_rmse()
    hybrid_rmse = evaluate_hybrid_rmse()

    results_df = pd.DataFrame(
        [
            {
                "Model": "Collaborative Filtering",
                "Precision@10": cf_precision,
                "Recall@10": cf_recall,
                "F1@10": cf_f1,
                "MSE": round(cf_rmse**2, 4),
                "RMSE": cf_rmse,
            },
            {
                "Model": "Content-Based Filtering",
                "Precision@10": cb_precision,
                "Recall@10": cb_recall,
                "F1@10": cb_f1,
                "MSE": round(cb_rmse**2, 4),
                "RMSE": cb_rmse,
            },
            {
                "Model": "Hybrid Filtering",
                "Precision@10": hybrid_precision,
                "Recall@10": hybrid_recall,
                "F1@10": hybrid_f1,
                "MSE": round(hybrid_rmse**2, 4),
                "RMSE": hybrid_rmse,
            },
        ]
    )
    results_df.to_csv(EVALUATION_CSV, index=False)
    return results_df
