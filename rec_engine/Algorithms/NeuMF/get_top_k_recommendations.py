import pandas as pd
import torch

def get_top_k_recommendations(model, user_id, user_to_idx, item_ids, item_to_idx, device, k=10, rated_items=None):
    """
    Get top-K recommendations for a user
    
    Args:
        model: Trained NeuMF model
        user_id: ID of the user to recommend for
        user_to_idx: User ID to index mapping
        item_ids: List of all item IDs in the dataset
        item_to_idx: Item ID to index mapping
        device: Device to run computations on
        k: Number of recommendations to return
        rated_items: Set of items the user has already rated (to exclude from recommendations)
    """
    model.eval()
    
    # Convert user ID to index
    if isinstance(user_id, str):
        user_idx = torch.LongTensor([user_to_idx[user_id]])
    else:
        user_idx = torch.LongTensor([user_id])
    user_idx = user_idx.to(device)
    
    # Prepare all item indices
    all_item_indices = torch.LongTensor([item_to_idx[item] for item in item_ids]).to(device)
    
    # Create user tensor with same length as items (for batch prediction)
    user_indices = user_idx.repeat(len(all_item_indices))
    
    # Predict ratings for all items
    with torch.no_grad():
        predictions = model(user_indices, all_item_indices)
        predictions = torch.clamp(predictions, min=0.5, max=5.0)  # Clip to rating range
    
    # Convert predictions to numpy array
    predictions = predictions.cpu().numpy()
    
    # Create a dictionary of item IDs to predicted ratings
    item_ratings = {item_id: pred for item_id, pred in zip(item_ids, predictions)}
    
    # Filter out items the user has already rated if provided
    if rated_items is not None:
        item_ratings = {item: rating for item, rating in item_ratings.items() 
                       if item not in rated_items}
    
    # Sort items by predicted rating
    sorted_ratings = sorted(item_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-K items
    top_k = sorted_ratings[:k]
    
    return top_k
