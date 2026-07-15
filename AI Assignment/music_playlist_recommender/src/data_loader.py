from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SONGS_CSV = DATA_DIR / "songs.csv"
INTERACTIONS_CSV = DATA_DIR / "user_interactions.csv"

SONG_COLUMNS = [
    "song_id",
    "title",
    "artist",
    "genre",
    "tempo",
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "popularity",
]

INTERACTION_COLUMNS = ["user_id", "song_id", "rating", "play_count", "liked"]


def create_sample_csv_data():
    """Create realistic CSV sample data if the project has no CSV database yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    genres = ["Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "R&B", "Country"]
    artists = [
        "Ava Stone",
        "Neon Rivers",
        "Midnight Echo",
        "Luna Park",
        "The Velvet Keys",
        "DJ Horizon",
        "Maya Blue",
        "Northline",
        "Golden Avenue",
        "Kai Morgan",
        "Silver Pulse",
        "Rose Harbor",
        "Urban Meadow",
        "Nova Beats",
        "The Calm Lights",
    ]
    title_words = [
        "Dream",
        "Night",
        "Motion",
        "Signal",
        "Heart",
        "Sky",
        "Road",
        "Fire",
        "Ocean",
        "City",
        "Light",
        "Echo",
        "Summer",
        "Velvet",
        "Gravity",
        "River",
        "Moon",
        "Golden",
        "Memory",
        "Rhythm",
    ]

    songs = []
    for song_number in range(1, 101):
        genre = rng.choice(genres)
        artist = rng.choice(artists)
        title = f"{rng.choice(title_words)} {rng.choice(title_words)} {song_number}"

        # Audio features are stored directly in CSV columns, so no external database is needed.
        songs.append(
            {
                "song_id": f"S{song_number:03d}",
                "title": title,
                "artist": artist,
                "genre": genre,
                "tempo": int(rng.integers(70, 181)),
                "energy": round(float(rng.uniform(0.20, 0.98)), 2),
                "danceability": round(float(rng.uniform(0.15, 0.95)), 2),
                "valence": round(float(rng.uniform(0.10, 0.95)), 2),
                "acousticness": round(float(rng.uniform(0.02, 0.90)), 2),
                "popularity": int(rng.integers(20, 101)),
            }
        )

    songs_df = pd.DataFrame(songs, columns=SONG_COLUMNS)

    interactions = []
    user_ids = [f"U{user_number:03d}" for user_number in range(1, 31)]

    for user_id in user_ids:
        # Each user listens to a different sample of songs.
        listened_song_ids = rng.choice(songs_df["song_id"], size=18, replace=False)
        for song_id in listened_song_ids:
            song = songs_df.loc[songs_df["song_id"] == song_id].iloc[0]
            genre_bonus = 0.6 if song["genre"] in rng.choice(genres, size=2, replace=False) else 0
            popularity_bonus = song["popularity"] / 100
            rating = float(np.clip(rng.normal(3.0 + genre_bonus + popularity_bonus, 0.9), 1, 5))
            rating = round(rating, 1)
            play_count = int(max(1, rng.poisson(lam=rating * 2)))
            interactions.append(
                {
                    "user_id": user_id,
                    "song_id": song_id,
                    "rating": rating,
                    "play_count": play_count,
                    "liked": 1 if rating >= 4 else 0,
                }
            )

    interactions_df = pd.DataFrame(interactions, columns=INTERACTION_COLUMNS)

    # Guarantee at least 500 rows even if the generator is edited later.
    if len(interactions_df) < 500:
        extra_needed = 500 - len(interactions_df)
        extra_rows = interactions_df.sample(extra_needed, replace=True, random_state=42)
        interactions_df = pd.concat([interactions_df, extra_rows], ignore_index=True)

    songs_df.to_csv(SONGS_CSV, index=False)
    interactions_df.to_csv(INTERACTIONS_CSV, index=False)


def validate_csv_columns(dataframe, required_columns, file_name):
    """Check that a CSV file has all columns required by the assignment."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{file_name} is missing these columns: {missing_columns}")


def clean_csv_data(songs_df, interactions_df, save=True):
    """Remove duplicates, fill simple missing values, and optionally save cleaned CSV files."""
    songs_df = songs_df.drop_duplicates(subset=["song_id"]).copy()
    interactions_df = interactions_df.drop_duplicates(subset=["user_id", "song_id"]).copy()

    for column in ["title", "artist", "genre"]:
        songs_df[column] = songs_df[column].fillna("Unknown")

    numeric_song_columns = ["tempo", "energy", "danceability", "valence", "acousticness", "popularity"]
    for column in numeric_song_columns:
        songs_df[column] = pd.to_numeric(songs_df[column], errors="coerce")
        songs_df[column] = songs_df[column].fillna(songs_df[column].median())

    interactions_df["rating"] = pd.to_numeric(interactions_df["rating"], errors="coerce").fillna(3)
    interactions_df["play_count"] = pd.to_numeric(interactions_df["play_count"], errors="coerce").fillna(1)
    interactions_df["liked"] = pd.to_numeric(interactions_df["liked"], errors="coerce").fillna(0).astype(int)
    interactions_df["rating"] = interactions_df["rating"].clip(1, 5)
    interactions_df["play_count"] = interactions_df["play_count"].clip(lower=0).astype(int)

    if save:
        songs_df.to_csv(SONGS_CSV, index=False)
        interactions_df.to_csv(INTERACTIONS_CSV, index=False)

    return songs_df, interactions_df


def load_songs_csv():
    """Load songs.csv using pandas and generate sample data if the CSV is missing."""
    if not SONGS_CSV.exists() or not INTERACTIONS_CSV.exists():
        create_sample_csv_data()

    songs_df = pd.read_csv(SONGS_CSV)
    validate_csv_columns(songs_df, SONG_COLUMNS, "songs.csv")
    return songs_df


def load_user_interactions_csv():
    """Load user_interactions.csv using pandas and generate sample data if missing."""
    if not SONGS_CSV.exists() or not INTERACTIONS_CSV.exists():
        create_sample_csv_data()

    interactions_df = pd.read_csv(INTERACTIONS_CSV)
    validate_csv_columns(interactions_df, INTERACTION_COLUMNS, "user_interactions.csv")
    return interactions_df


def load_clean_data():
    """Convenience function used by the app and recommender modules."""
    songs_df = load_songs_csv()
    interactions_df = load_user_interactions_csv()
    return clean_csv_data(songs_df, interactions_df)
