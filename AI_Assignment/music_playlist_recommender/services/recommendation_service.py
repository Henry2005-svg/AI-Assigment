import pandas as pd

from src.content_based_filtering import recommend_similar_songs

# create a recommendation service class
class RecommendationService:

    def get_content_recommendation(self, song_id, top_n=10):

        recommendations = recommend_similar_songs(song_id, top_n)

        # return the recommendations as a list of dictionaries
        return recommendations.to_dict(orient="records")