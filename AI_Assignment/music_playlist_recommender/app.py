# create backend API for the music playlist recommender
from flask import Flask
# create a blueprint for the recommendation routes
from routes.recommend_routes import recommend_bp

# create backend API for the music playlist recommender
app = Flask(__name__)
app.register_blueprint(recommend_bp)


# when the user visits the root URL, return a welcome message
@app.route('/')
def home():
    return {
        "message": "Music Recommendation Backend Running!"
    }

if __name__ == '__main__':
    app.run(debug=True)
