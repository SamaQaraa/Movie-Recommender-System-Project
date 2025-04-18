import os
import sys
import torch
import pandas as pd
from .load_model import load_model_and_mappings
from .get_top_k_recommendations import get_top_k_recommendations
from .inference import NeuMF

def get_user_rated_items(user_id, ratings_path):
    """Get the set of items that a user has already rated"""
    df = pd.read_csv(ratings_path)
    return set(df[df["userId"] == user_id]["movieId"].values)

def get_movie_titles(movies_path):
    """Get a mapping of movie IDs to titles"""
    df = pd.read_csv(movies_path)
    return dict(zip(df["movieId"], df["title"]))

def main():
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path = "D:/ITI/Rec_Sys_Intake_45/project_descrption/project"
    
    # Add the project root to the Python path
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)
    
    # Paths to data files
    data_dir = os.path.join(workspace_path, 'data', 'ml-latest-small')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    # Load the model and mappings
    print("Loading model and mappings...")
    wrapped_model, user_to_idx, item_to_idx = load_model_and_mappings(
        model_class=NeuMF,
        path=os.path.join(current_dir, 'recommender_model/'),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mf_dim=16,
        mlp_dim=64,
        layers=[128, 64, 32]
    )
    
    # Get all item IDs
    item_ids = list(item_to_idx.keys())
    
    # Get movie titles
    movie_titles = get_movie_titles(movies_path)
    
    # Make predictions for user 80
    user_id = 80
    print(f"\nGetting recommendations for user {user_id}...")
    
    # Get items the user has already rated
    rated_items = get_user_rated_items(user_id, ratings_path)
    
    # Get top 10 recommendations
    recommendations = get_top_k_recommendations(
        model=wrapped_model.model,
        user_id=user_id,
        user_to_idx=user_to_idx,
        item_ids=item_ids,
        item_to_idx=item_to_idx,
        device=wrapped_model.device,
        k=10,
        rated_items=rated_items
    )
    
    # Print the recommendations
    print(f"\nTop 10 recommendations for user {user_id}:")
    for i, (item_id, rating) in enumerate(recommendations, 1):
        movie_title = movie_titles.get(item_id, "Unknown Movie")
        rating_float = float(rating)  # Convert numpy array to float
        print(f"{i}. {movie_title} (ID: {item_id}) - Predicted rating: {rating_float*4.5 + 0.5:.2f}")

if __name__ == "__main__":
    main() 
