import os
from .hyper_predictor import HyperPredictor

def main():
    # Get the absolute path to the data file
    workspace_path = "D:/ITI/Rec_Sys_Intake_45/project_descrption/project"
    data_path = os.path.join(workspace_path, 'data', 'ml-latest-small', 'ratings.csv')
    
    # Initialize hyper-predictor
    print("Initializing hyper-predictor...")
    predictor = HyperPredictor(data_path)
    
    # Get movie titles
    movie_titles = predictor.get_movie_titles()
    
    # Make predictions for user 80
    user_id = 80
    print(f"\nGetting recommendations for user {user_id}...")
    
    # Get top 10 recommendations
    recommendations = predictor.get_top_k_recommendations(user_id, k=10)
    
    # Print the recommendations
    print(f"\nTop 10 recommendations for user {user_id}:")
    for i, (item_id, rating) in enumerate(recommendations, 1):
        movie_title = movie_titles.get(item_id, "Unknown Movie")
        print(f"{i}. {movie_title} (ID: {item_id}) - Predicted rating: {rating:.2f}")

if __name__ == "__main__":
    main() 
