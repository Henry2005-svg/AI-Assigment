from flask import Blueprint, jsonify, request

from services.recommendation_service import RecommendationService

recommend_bp = Blueprint("recommend", __name__)

recommendation_service = RecommendationService()


@recommend_bp.route("/recommend/content", methods=["GET"])
def recommend_content():

    title = request.args.get("title")

    if not title:
        return jsonify({
            "status": "error",
            "message": "Please provide a song title."
        }), 400

    recommendations = recommendation_service.get_content_recommendation(
        title=title,
        top_n=10
    )

    return jsonify(recommendations)