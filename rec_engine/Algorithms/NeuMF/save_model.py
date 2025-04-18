import os
import torch
import pickle

def save_model_and_mappings(model, user_to_idx, item_to_idx, path='recommender_model/'):
    """
    Save the model and ID mappings for future use.
    
    Args:
        model: Trained PyTorch model
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        path: Directory to save files
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(path, 'model_state_dict.pth'))
    
    # Save mappings
    with open(os.path.join(path, 'mappings.pkl'), 'wb') as f:
        pickle.dump({
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'num_users': len(user_to_idx),
            'num_items': len(item_to_idx)
        }, f)
    
    print(f"Model and mappings saved to {path}")
