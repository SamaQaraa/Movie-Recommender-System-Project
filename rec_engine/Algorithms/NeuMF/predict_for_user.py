import torch
import os
import sys
import pandas as pd
# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the modules
try:
    from rec_engine.Algorithms.NeuMF.load_model import load_model_and_mappings
    from rec_engine.Algorithms.NeuMF.get_top_k_recommendations import get_top_k_recommendations
    from rec_engine.Algorithms.NeuMF.train import NeuMF
except ImportError:
    # If imports fail, try importing directly from the current directory
    sys.path.insert(0, os.path.dirname(current_dir))
    from NeuMF.load_model import load_model_and_mappings
    from NeuMF.get_top_k_recommendations import get_top_k_recommendations
    from NeuMF.train import NeuMF

def predict_for_user(user_id=80, k=10):
    """
    Load the model and make predictions for a specific user.
    
    Args:
        user_id: The user ID to make predictions for
        k: Number of top recommendations to return
        
    Returns:
        List of top k recommended items with their predicted ratings
    """
    # Load the model and mappings
    wrapped_model, user_to_idx, item_to_idx = load_model_and_mappings(
        model_class=NeuMF,
        path=os.path.join(current_dir, 'recommender_model/'),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mf_dim=16,
        mlp_dim=64,
        layers=[128, 64, 32]
    )
    
    # Get top k recommendations for the user
    recommendations = get_top_k_recommendations(
        model=wrapped_model,
        user_id=user_id,
        k=k,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx
    )
    
    return recommendations

if __name__ == "__main__":
    # Make predictions for user 80
    print("Loading model and making predictions...")
    recommendations = predict_for_user(user_id=80, k=10)
    
    # Print the recommendations
    print(f"\nTop 10 recommendations for user 80:")
    for i, (item_id, rating) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}: Predicted rating {rating:.2f}") 
