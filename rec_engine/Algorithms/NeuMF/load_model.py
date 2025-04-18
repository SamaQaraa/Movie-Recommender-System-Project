import os
import torch
import pickle
from .PyTorchWrapper import PyTorchWrapper

def load_model_and_mappings(model_class, path='recommender_model/', device='cpu', **model_params):
    """
    Load model and mappings for prediction.
    
    Args:
        model_class: The model class (architecture)
        path: Directory containing saved files
        device: Target device
        model_params: Additional model parameters
        
    Returns:
        wrapped_model: PyTorchWrapper instance containing the loaded model
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
    """
    # Load mappings
    with open(os.path.join(path, 'mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    
    # Initialize model with saved dimensions
    model = model_class(
        num_users=mappings['num_users'],
        num_items=mappings['num_items'],
        **model_params
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(path, 'model_state_dict.pth'), map_location=device))
    model.eval()
    
    # Create wrapped model
    wrapped_model = PyTorchWrapper(model, mappings['user_to_idx'], mappings['item_to_idx'], device)
    
    return wrapped_model, mappings['user_to_idx'], mappings['item_to_idx']
