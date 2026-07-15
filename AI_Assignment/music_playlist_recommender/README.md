# Music Playlist Recommender System

This is a beginner-friendly Artificial Intelligence assignment prototype built with Python. It recommends songs using three approaches:

1. Collaborative Filtering
2. Content-Based Filtering
3. Hybrid Filtering

The project uses CSV files as the data source. It does not use MySQL, SQLite, Firebase, MongoDB, online APIs, paid APIs, or secret keys.

## Project Objective

The objective is to build a simple music recommender system that suggests songs to users based on:

- user listening behaviour
- ratings and play counts
- liked songs
- song metadata
- audio features such as genre, artist, tempo, energy, danceability, valence, acousticness, and popularity

## Why CSV Files Are Used

CSV data storage means the system reads and writes data from `.csv` files instead of a database server. This is useful for a beginner assignment because CSV files are easy to open, inspect, edit, and explain during a presentation.

The main CSV files are:

- `data/songs.csv`
- `data/user_interactions.csv`

If these files are missing, the system automatically creates realistic sample data.

## Project Structure

```text
music_playlist_recommender/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- songs.csv
|   `-- user_interactions.csv
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- collaborative_filtering.py
|   |-- content_based_filtering.py
|   |-- hybrid_filtering.py
|   |-- evaluation.py
|   `-- utils.py
`-- outputs/
    |-- evaluation_results.csv
    |-- processed_songs.csv
    `-- processed_user_interactions.csv
```

## Dataset Structure

### `songs.csv`

| Column | Description |
| --- | --- |
| `song_id` | Unique song ID |
| `title` | Song title |
| `artist` | Artist name |
| `genre` | Song genre |
| `tempo` | Speed of the song in beats per minute |
| `energy` | Energy level from 0 to 1 |
| `danceability` | Danceability level from 0 to 1 |
| `valence` | Mood or positivity level from 0 to 1 |
| `acousticness` | Acoustic sound level from 0 to 1 |
| `popularity` | Popularity score from 0 to 100 |

### `user_interactions.csv`

| Column | Description |
| --- | --- |
| `user_id` | Unique user ID |
| `song_id` | Song listened to by the user |
| `rating` | User rating from 1 to 5 |
| `play_count` | Number of times the user played the song |
| `liked` | 1 if the user liked the song, 0 otherwise |

## Installation

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

The app includes these pages:

- Dataset Preview
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Filtering
- Model Evaluation

## How Collaborative Filtering Works

Collaborative filtering recommends songs by comparing users. If User A and User B have similar ratings, the system can recommend songs liked by User B to User A.

This project:

- creates a user-song rating matrix
- calculates cosine similarity between users
- predicts ratings for songs the selected user has not rated
- recommends the songs with the highest predicted ratings

Cosine similarity measures how similar two users or songs are based on their data pattern.

## How Content-Based Filtering Works

Content-based filtering recommends songs similar to a selected song. It uses song details and audio features such as genre, artist, tempo, energy, danceability, valence, acousticness, and popularity.

This project:

- one-hot encodes genre and artist
- scales numerical audio features
- calculates cosine similarity between songs
- recommends songs with the highest similarity score

## How Hybrid Filtering Works

Hybrid filtering combines collaborative filtering and content-based filtering.

Default formula:

```text
Hybrid Score = 0.6 * Collaborative Filtering Score + 0.4 * Content-Based Score
```

The Streamlit app allows the user to adjust the collaborative filtering weight. The content-based weight is calculated automatically.

## Evaluation

The system evaluates the recommenders using:

- Precision@10
- Recall@10
- F1@10
- MSE
- RMSE

Precision@K means: out of the top K recommendations, how many were relevant?

Recall@K means: out of all relevant songs, how many did the system recommend?

F1@K combines precision and recall into one balanced score.

MSE means Mean Squared Error. It measures rating prediction error.

RMSE means Root Mean Squared Error. It is easier to interpret because it uses the same scale as ratings.

Evaluation results are saved to:

```text
outputs/evaluation_results.csv
```

## Expected Demo Flow

1. Run the Streamlit app.
2. Open Dataset Preview.
3. Show `songs.csv` and `user_interactions.csv`.
4. Open Collaborative Filtering.
5. Select a user and generate recommendations.
6. Open Content-Based Filtering.
7. Select a song and generate similar songs.
8. Open Hybrid Filtering.
9. Select a user and adjust the recommendation weight.
10. Open Model Evaluation.
11. Compare all three models.

## Limitations

- The dataset is sample data, not real streaming platform data.
- The recommender uses simple machine learning methods only.
- The system does not include login or user account management.
- The content-based RMSE is a simple estimated metric for assignment comparison.
- Recommendations may be less accurate than production music platforms.

## Future Improvements

- Add more songs and users.
- Collect real user feedback from a questionnaire.
- Add playlist creation and export features.
- Improve evaluation with train-test splitting for all models.
- Add charts for genre popularity and recommendation performance.
