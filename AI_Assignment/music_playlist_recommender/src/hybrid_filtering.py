import numpy as np
import pandas as pd

from src.collaborative_filtering import calculate_user_similarity, predict_rating
from src.content_based_filtering import calculate_song_similarity, prepare_song_feature_matrix
from src.data_loader import load_clean_data
from src.ml_helpers import mean_squared_error, train_test_split
from src.preprocessing import create_user_song_matrix


def get_collaborative_score(user_id, song_id):
    """Return the collaborative filtering score for one user-song pair."""
    _, interactions_df = load_clean_data()
    user_song_matrix = create_user_song_matrix(interactions_df)
    user_similarity_df = calculate_user_similarity(user_song_matrix)
    return predict_rating(user_id, song_id, user_song_matrix, user_similarity_df)


def get_content_score(user_id, song_id):
    """Estimate content score by comparing a song with songs the user already liked."""
    songs_df, interactions_df = load_clean_data()
    liked_song_ids = interactions_df[
        (interactions_df["user_id"] == user_id) & (interactions_df["liked"] == 1)
    ]["song_id"].tolist()

    if song_id not in songs_df["song_id"].values or not liked_song_ids:
        return 0

    feature_matrix, _ = prepare_song_feature_matrix(songs_df)
    song_similarity = calculate_song_similarity(feature_matrix)
    target_index = songs_df.index[songs_df["song_id"] == song_id][0]

    liked_indexes = songs_df[songs_df["song_id"].isin(liked_song_ids)].index.tolist()
    scores = [song_similarity[target_index][liked_index] for liked_index in liked_indexes if liked_index != target_index]

    if not scores:
        return 0

    # Convert cosine similarity from 0-1 style score to the 1-5 rating scale.
    return round(float(1 + 4 * np.mean(scores)), 3)


def recommend_songs_hybrid(user_id, top_n=10, cf_weight=0.6, content_weight=0.4, exclude_seen=True):
    """Combine collaborative and content-based scores into one hybrid playlist."""
    songs_df, interactions_df = load_clean_data()
    user_song_matrix = create_user_song_matrix(interactions_df)
    user_similarity_df = calculate_user_similarity(user_song_matrix)
    feature_matrix, _ = prepare_song_feature_matrix(songs_df)
    song_similarity = calculate_song_similarity(feature_matrix)

    if user_id not in user_song_matrix.index:
        return pd.DataFrame()

    liked_song_ids = interactions_df[
        (interactions_df["user_id"] == user_id) & (interactions_df["liked"] == 1)
    ]["song_id"].tolist()
    liked_indexes = songs_df[songs_df["song_id"].isin(liked_song_ids)].index.tolist()
    already_rated = set(interactions_df.loc[interactions_df["user_id"] == user_id, "song_id"])

    recommendations = []
    for _, song in songs_df.iterrows():
        song_id = song["song_id"]
        if exclude_seen and song_id in already_rated:
            continue

        collaborative_score = predict_rating(user_id, song_id, user_song_matrix, user_similarity_df)
        target_index = songs_df.index[songs_df["song_id"] == song_id][0]

        if liked_indexes:
            content_raw = np.mean([song_similarity[target_index][liked_index] for liked_index in liked_indexes])
            content_score = 1 + 4 * content_raw
        else:
            content_score = song["popularity"] / 20

        if collaborative_score == 0:
            collaborative_score = interactions_df["rating"].mean()

        hybrid_score = (cf_weight * collaborative_score) + (content_weight * content_score)
        recommendations.append(
            {
                "song_id": song_id,
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "collaborative_score": round(float(collaborative_score), 3),
                "content_score": round(float(content_score), 3),
                "hybrid_score": round(float(hybrid_score), 3),
                "reason": "Recommended because similar users liked it and the song features match your listening preference.",
            }
        )

    return pd.DataFrame(recommendations).sort_values("hybrid_score", ascending=False).head(top_n)


def evaluate_hybrid_rmse(cf_weight=0.6, content_weight=0.4):
    """Evaluate hybrid rating predictions with RMSE."""
    songs_df, interactions_df = load_clean_data()
    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
    user_song_matrix = create_user_song_matrix(train_df)
    user_similarity_df = calculate_user_similarity(user_song_matrix)
    feature_matrix, _ = prepare_song_feature_matrix(songs_df)
    song_similarity = calculate_song_similarity(feature_matrix)
    global_mean = train_df["rating"].mean()

    actual_ratings = []
    predicted_ratings = []

    for _, row in test_df.iterrows():
        collaborative_score = predict_rating(row["user_id"], row["song_id"], user_song_matrix, user_similarity_df)
        if collaborative_score == 0:
            collaborative_score = global_mean

        liked_song_ids = train_df[(train_df["user_id"] == row["user_id"]) & (train_df["liked"] == 1)]["song_id"]
        target_indexes = songs_df.index[songs_df["song_id"] == row["song_id"]].tolist()
        liked_indexes = songs_df[songs_df["song_id"].isin(liked_song_ids)].index.tolist()

        if target_indexes and liked_indexes:
            content_raw = np.mean([song_similarity[target_indexes[0]][liked_index] for liked_index in liked_indexes])
            content_score = 1 + 4 * content_raw
        else:
            content_score = global_mean

        predicted = (cf_weight * collaborative_score) + (content_weight * content_score)
        actual_ratings.append(row["rating"])
        predicted_ratings.append(predicted)

    mse = mean_squared_error(actual_ratings, predicted_ratings)
    return round(float(np.sqrt(mse)), 4)


def compare_all_models():
    """Shortcut used by the Streamlit app to create the comparison table."""
    from src.evaluation import evaluate_all_models

    return evaluate_all_models()
