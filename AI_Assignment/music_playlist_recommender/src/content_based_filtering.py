import numpy as np
import pandas as pd

from src.data_loader import load_clean_data
from src.ml_helpers import cosine_similarity
from src.preprocessing import preprocess_song_features


def prepare_song_feature_matrix(songs_df):
    """Prepare genre, artist, and audio features for content-based filtering."""
    feature_matrix, transformer = preprocess_song_features(songs_df)
    return feature_matrix, transformer


def calculate_song_similarity(feature_matrix):
    """Find songs with similar genre, artist, tempo, energy, danceability, and mood."""
    return cosine_similarity(feature_matrix)


def recommend_similar_songs(song_id, top_n=10):
    """Recommend songs similar to a selected song."""
    songs_df, _ = load_clean_data()
    feature_matrix, _ = prepare_song_feature_matrix(songs_df)
    song_similarity = calculate_song_similarity(feature_matrix)

    if song_id not in songs_df["song_id"].values:
        return pd.DataFrame()

    song_index = songs_df.index[songs_df["song_id"] == song_id][0]
    similarity_scores = list(enumerate(song_similarity[song_index]))
    similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

    recommendations = []
    for index, score in similarity_scores:
        if index == song_index:
            continue
        song = songs_df.iloc[index]
        recommendations.append(
            {
                "song_id": song["song_id"],
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "similarity_score": round(float(score), 3),
                "reason": "Recommended because this song has similar genre, tempo, energy, danceability, and mood.",
            }
        )
        if len(recommendations) == top_n:
            break

    return pd.DataFrame(recommendations)


def recommend_by_preferences(genre=None, energy_level=None, top_n=10):
    """Recommend popular songs that match a simple genre or energy preference."""
    songs_df, _ = load_clean_data()
    filtered_df = songs_df.copy()

    if genre:
        filtered_df = filtered_df[filtered_df["genre"] == genre]

    if energy_level == "Low":
        filtered_df = filtered_df[filtered_df["energy"] < 0.40]
    elif energy_level == "Medium":
        filtered_df = filtered_df[(filtered_df["energy"] >= 0.40) & (filtered_df["energy"] < 0.70)]
    elif energy_level == "High":
        filtered_df = filtered_df[filtered_df["energy"] >= 0.70]

    filtered_df = filtered_df.sort_values(["popularity", "danceability"], ascending=False).head(top_n).copy()
    filtered_df["similarity_score"] = filtered_df["popularity"] / 100
    filtered_df["reason"] = "Recommended because it matches your selected music preferences."
    return filtered_df[["song_id", "title", "artist", "genre", "similarity_score", "reason"]]


def evaluate_content_precision(top_k=10):
    """Precision@K checks how many recommended songs are relevant in the top K results."""
    songs_df, interactions_df = load_clean_data()
    liked_df = interactions_df[interactions_df["liked"] == 1]

    precision_scores = []
    for user_id in liked_df["user_id"].unique():
        user_liked_songs = liked_df[liked_df["user_id"] == user_id]["song_id"].tolist()
        if len(user_liked_songs) < 2:
            continue

        seed_song = user_liked_songs[0]
        recommendations = recommend_similar_songs(seed_song, top_n=top_k)
        recommended_ids = set(recommendations["song_id"])
        relevant_ids = set(user_liked_songs[1:])

        if recommended_ids:
            precision_scores.append(len(recommended_ids & relevant_ids) / len(recommended_ids))

    if not precision_scores:
        return 0

    return round(float(np.mean(precision_scores)), 4)
