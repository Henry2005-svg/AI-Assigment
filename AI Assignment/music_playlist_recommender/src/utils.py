def format_dataframe_for_display(dataframe):
    """Keep Streamlit tables tidy by resetting the row numbers."""
    if dataframe is None or dataframe.empty:
        return dataframe
    return dataframe.reset_index(drop=True)


def explain_metric(metric_name):
    """Simple explanations suitable for a beginner presentation."""
    explanations = {
        "Precision@10": "Out of the top 10 recommended songs, the percentage that were relevant.",
        "Recall@10": "Out of all songs the user liked, the percentage found by the recommender.",
        "F1@10": "A balanced score that combines Precision@10 and Recall@10.",
        "MSE": "Mean Squared Error. It measures rating prediction error, and lower is better.",
        "RMSE": "Root Mean Squared Error. It is easier to read because it uses the same scale as ratings.",
    }
    return explanations.get(metric_name, "No explanation available.")
