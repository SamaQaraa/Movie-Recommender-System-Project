# server.py
"""
FastAPI server implementation for the movie recommendation system.
This server provides REST API endpoints for user management and movie recommendations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from typing import List, Dict, Optional
import os
import sys

# Add the project root to Python path
workspace_path = "D:/ITI/Rec_Sys_Intake_45/project_descrption/project"
sys.path.append(workspace_path)

from rec_engine.Algorithms.Hybrid.hyper_predictor import HyperPredictor

# Initialize FastAPI application
app = FastAPI(title="Movie Recommendation API")

# Define data models for request/response validation
class MovieList(BaseModel):
    movies: List[str]

class RecommendationRequest(BaseModel):
    user_id: str
    count: int

# Initialize data paths and load datasets
data_path = os.path.join(workspace_path, 'data', 'ml-latest-small', 'ratings.csv')
movies_path = os.path.join(workspace_path, 'data', 'ml-latest-small', 'movies.csv')

# Load ratings and movies data
ratings_df = pd.read_csv(data_path)
movies_df = pd.read_csv(movies_path)

# Get unique user IDs and convert to strings
users_ids = ratings_df["userId"].unique().tolist()
users_ids = [str(i) for i in users_ids]

# Create user_movies_db from actual ratings for quick lookup
user_movies_db = {}
for user_id in users_ids:
    user_ratings = ratings_df[ratings_df["userId"] == int(user_id)]
    user_movies = user_ratings.merge(movies_df, on="movieId")["title"].tolist()
    user_movies_db[user_id] = user_movies

# Initialize hyper-predictor for recommendations
print("Initializing hyper-predictor...")
try:
    predictor = HyperPredictor(data_path)
    print("Hyper-predictor initialized successfully")
except Exception as e:
    print(f"Error initializing hyper-predictor: {str(e)}")
    raise

# API endpoints
@app.get("/")
async def root():
    """Check if the API is running"""
    return {"message": "Movie Recommendation API is running"}

@app.get("/users")
async def get_users():
    """Get list of all available user IDs"""
    return {"users": users_ids}

@app.get("/user/{user_id}/movies", response_model=MovieList)
async def get_user_movies(user_id: str):
    """Get watched movies for a specific user"""
    if user_id not in user_movies_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return {"movies": user_movies_db[user_id]}

@app.post("/recommendations", response_model=MovieList)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    user_id = request.user_id
    count = request.count
    
    if user_id not in users_ids:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    try:
        # Get recommendations from hyper-predictor
        recommendations = predictor.get_top_k_recommendations(int(user_id), k=count)
        
        # Get movie titles
        movie_titles = predictor.get_movie_titles()
        recommended_movies = [movie_titles.get(item_id, "Unknown Movie") for item_id, _ in recommendations]
        
        if not recommended_movies:
            raise HTTPException(status_code=404, detail="No recommendations available")
            
        return {"movies": recommended_movies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
