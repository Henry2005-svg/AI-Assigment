import streamlit as st

from src.collaborative_filtering import recommend_songs_collaborative
from src.content_based_filtering import recommend_similar_songs
from src.data_loader import load_clean_data
from src.evaluation import EVALUATION_CSV, evaluate_all_models
from src.hybrid_filtering import recommend_songs_hybrid
from src.preprocessing import save_processed_data_to_csv
from src.utils import explain_metric, format_dataframe_for_display
## niga

st.set_page_config(page_title="Music Playlist Recommender", layout="wide")


@st.cache_data
def get_data():
    songs_df, interactions_df = load_clean_data()
    save_processed_data_to_csv(songs_df, interactions_df)
    return songs_df, interactions_df


songs_df, interactions_df = get_data()

st.title("Music Playlist Recommender System")
st.write(
    "A beginner-friendly AI assignment prototype that recommends music using "
    "Collaborative Filtering, Content-Based Filtering, and Hybrid Filtering. "
    "All data is stored in CSV files only."
)

menu = st.sidebar.radio(
    "Sidebar Menu",
    [
        "Dataset Preview",
        "Collaborative Filtering",
        "Content-Based Filtering",
        "Hybrid Filtering",
        "Model Evaluation",
    ],
)

if menu == "Dataset Preview":
    st.header("CSV Dataset Preview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Songs", len(songs_df))
    col2.metric("Total Users", interactions_df["user_id"].nunique())
    col3.metric("Total Interactions", len(interactions_df))

    st.subheader("songs.csv")
    st.dataframe(songs_df, use_container_width=True)

    st.subheader("user_interactions.csv")
    st.dataframe(interactions_df, use_container_width=True)

elif menu == "Collaborative Filtering":
    st.header("Collaborative Filtering Recommendations")
    st.write(
        "Collaborative filtering recommends songs by finding users with similar listening "
        "behaviour and ratings."
    )

    user_id = st.selectbox("Select user_id", sorted(interactions_df["user_id"].unique()))
    top_n = st.slider("Number of recommended songs", 5, 20, 10)

    recommendations = recommend_songs_collaborative(user_id, top_n=top_n)
    st.subheader("Top Recommended Songs")
    st.dataframe(format_dataframe_for_display(recommendations), use_container_width=True)

elif menu == "Content-Based Filtering":
    st.header("Content-Based Filtering Recommendations")
    st.write(
        "Content-based filtering recommends songs with similar genre, artist, tempo, "
        "energy, danceability, mood, acousticness, and popularity."
    )

    title_to_id = dict(zip(songs_df["title"], songs_df["song_id"]))
    selected_title = st.selectbox("Select a song title", sorted(title_to_id.keys()))
    top_n = st.slider("Number of similar songs", 5, 20, 10)

    recommendations = recommend_similar_songs(title_to_id[selected_title], top_n=top_n)
    st.subheader("Top Similar Songs")
    st.dataframe(format_dataframe_for_display(recommendations), use_container_width=True)

elif menu == "Hybrid Filtering":
    st.header("Hybrid Filtering Recommendations")
    st.write(
        "Hybrid filtering combines collaborative filtering and content-based filtering "
        "to create a more balanced recommendation."
    )

    user_id = st.selectbox("Select user_id", sorted(interactions_df["user_id"].unique()))
    cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6, 0.1)
    content_weight = round(1.0 - cf_weight, 1)
    top_n = st.slider("Number of hybrid recommendations", 5, 20, 10)

    st.info(f"Content-Based Weight is automatically set to {content_weight}.")
    recommendations = recommend_songs_hybrid(
        user_id,
        top_n=top_n,
        cf_weight=cf_weight,
        content_weight=content_weight,
    )

    st.subheader("Top Hybrid Playlist")
    st.dataframe(format_dataframe_for_display(recommendations), use_container_width=True)

elif menu == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("The table compares all three recommender models using common evaluation metrics.")

    with st.spinner("Calculating evaluation metrics..."):
        results_df = evaluate_all_models()

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)
    st.caption(f"Saved to {EVALUATION_CSV}")

    st.subheader("Simple Metric Explanations")
    for metric in ["Precision@10", "Recall@10", "F1@10", "MSE", "RMSE"]:
        st.markdown(f"**{metric}:** {explain_metric(metric)}")
#testing