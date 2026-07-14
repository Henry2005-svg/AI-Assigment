from pathlib import Path

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
except ModuleNotFoundError:
    ColumnTransformer = None
    MinMaxScaler = None
    OneHotEncoder = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NUMERICAL_FEATURES = ["tempo", "energy", "danceability", "valence", "acousticness", "popularity"]
CATEGORICAL_FEATURES = ["genre", "artist"]


def preprocess_song_features(songs_df):
    """Scale audio features and one-hot encode genre/artist for content-based filtering."""
    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    if ColumnTransformer is None:
        encoded_categories = pd.get_dummies(songs_df[CATEGORICAL_FEATURES], dtype=float)
        scaled_numbers = songs_df[NUMERICAL_FEATURES].copy()
        for column in NUMERICAL_FEATURES:
            minimum = scaled_numbers[column].min()
            maximum = scaled_numbers[column].max()
            if maximum == minimum:
                scaled_numbers[column] = 0
            else:
                scaled_numbers[column] = (scaled_numbers[column] - minimum) / (maximum - minimum)
        feature_matrix = pd.concat([encoded_categories, scaled_numbers], axis=1)
        return feature_matrix.to_numpy(), None

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", MinMaxScaler(), NUMERICAL_FEATURES),
        ]
    )

    feature_matrix = transformer.fit_transform(songs_df[feature_columns])
    return feature_matrix, transformer


def preprocess_user_interactions(interactions_df):
    """Make sure ratings, play counts, and liked values are in useful numeric ranges."""
    processed_df = interactions_df.copy()
    processed_df["rating"] = pd.to_numeric(processed_df["rating"], errors="coerce").fillna(3).clip(1, 5)
    processed_df["play_count"] = pd.to_numeric(processed_df["play_count"], errors="coerce").fillna(0).clip(lower=0)
    processed_df["liked"] = pd.to_numeric(processed_df["liked"], errors="coerce").fillna(0).astype(int)
    return processed_df


def create_user_song_matrix(interactions_df):
    """Create a table where rows are users, columns are songs, and values are ratings."""
    return interactions_df.pivot_table(index="user_id", columns="song_id", values="rating", fill_value=0)


def save_processed_data_to_csv(songs_df, interactions_df):
    """Save cleaned/processed copies so students can inspect the preprocessing output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    songs_df.to_csv(OUTPUT_DIR / "processed_songs.csv", index=False)
    interactions_df.to_csv(OUTPUT_DIR / "processed_user_interactions.csv", index=False)
