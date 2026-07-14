import numpy as np
import pandas as pd
#eason sane
from src.data_loader import load_clean_data
from src.ml_helpers import cosine_similarity, mean_squared_error, train_test_split 
from src.preprocessing import create_user_song_matrix


def calculate_user_similarity(user_song_matrix):
    """Cosine similarity measures how similar users are based on their rating patterns."""
    similarity = cosine_similarity(user_song_matrix)
    return pd.DataFrame(similarity, index=user_song_matrix.index, columns=user_song_matrix.index)


def predict_rating(user_id, song_id, user_song_matrix, user_similarity_df):
    """Predict how much a user may like a song using ratings from similar users."""
    if user_id not in user_song_matrix.index or song_id not in user_song_matrix.columns:
        return 0

    user_similarities = user_similarity_df.loc[user_id].drop(user_id)
    song_ratings = user_song_matrix[song_id].drop(user_id)
    rated_mask = song_ratings > 0

    if rated_mask.sum() == 0:
        return 0

    relevant_similarities = user_similarities[rated_mask]
    relevant_ratings = song_ratings[rated_mask]
    similarity_sum = relevant_similarities.abs().sum()

    if similarity_sum == 0:
        return float(relevant_ratings.mean())

    predicted_rating = np.dot(relevant_similarities, relevant_ratings) / similarity_sum
    return round(float(np.clip(predicted_rating, 1, 5)), 3)


def recommend_songs_collaborative(user_id, top_n=10, exclude_seen=True):
    """Recommend Top-N songs because similar users enjoyed those songs."""
    songs_df, interactions_df = load_clean_data()
    user_song_matrix = create_user_song_matrix(interactions_df)
    user_similarity_df = calculate_user_similarity(user_song_matrix)

    if user_id not in user_song_matrix.index:
        return pd.DataFrame()

    already_rated = set(interactions_df.loc[interactions_df["user_id"] == user_id, "song_id"])
    if exclude_seen:
        candidate_song_ids = [song_id for song_id in songs_df["song_id"] if song_id not in already_rated]
    else:
        candidate_song_ids = songs_df["song_id"].tolist()

    recommendations = []
    for song_id in candidate_song_ids:
        predicted_rating = predict_rating(user_id, song_id, user_song_matrix, user_similarity_df)
        if predicted_rating > 0:
            recommendations.append({"song_id": song_id, "predicted_rating": predicted_rating})

    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        return recommendations_df

    recommendations_df = recommendations_df.sort_values("predicted_rating", ascending=False).head(top_n)
    recommendations_df = recommendations_df.merge(songs_df, on="song_id", how="left")
    recommendations_df["reason"] = "Recommended because users with similar listening behaviour liked this song."

    return recommendations_df[["song_id", "title", "artist", "genre", "predicted_rating", "reason"]]


def evaluate_collaborative_rmse():
    """RMSE shows the average size of rating prediction errors; lower is better."""
    _, interactions_df = load_clean_data()

    if len(interactions_df) < 10:
        return 0

    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
    user_song_matrix = create_user_song_matrix(train_df)
    user_similarity_df = calculate_user_similarity(user_song_matrix)

    actual_ratings = []
    predicted_ratings = []
    global_mean = train_df["rating"].mean()

    for _, row in test_df.iterrows():
        predicted = predict_rating(row["user_id"], row["song_id"], user_song_matrix, user_similarity_df)
        if predicted == 0:
            predicted = global_mean
        actual_ratings.append(row["rating"])
        predicted_ratings.append(predicted)

    mse = mean_squared_error(actual_ratings, predicted_ratings)
    return round(float(np.sqrt(mse)), 4)
