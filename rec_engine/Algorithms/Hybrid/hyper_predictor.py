"""
Hybrid recommendation system that combines multiple models for better predictions.
This module implements a hyper-predictor that uses both SVD and NeuMF models.
"""

import os
import torch
import pandas as pd
import numpy as np
from surprise import SVD, SVDpp
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader
from ..NeuMF.inference import NeuMF
from ..NeuMF.load_model import load_model_and_mappings

class HyperPredictor:
    """Class that combines predictions from multiple recommendation models"""
    
    def __init__(self, data_path, model_path='recommender_model/'):
        """
        Initialize the hyper-predictor with multiple models
        
        Args:
            data_path: Path to the ratings.csv file
            model_path: Directory containing saved models
        """
        self.data_path = data_path
        self.model_path = model_path
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models"""
        # Initialize NeuMF
        self.neumf_model, self.user_to_idx, self.item_to_idx = load_model_and_mappings(
            model_class=NeuMF,
            path=os.path.join(os.path.dirname(__file__), '../NeuMF', self.model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            mf_dim=16,
            mlp_dim=64,
            layers=[128, 64, 32]
        )
        
        # Initialize Surprise models
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.df[['userId', 'movieId', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        
        self.svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        
        
        # Train Surprise models
        print("Training SVD...")
        self.svd.fit(trainset)
        
        
    def predict(self, user_id, item_id):
        """
        Get predictions from all models and combine them
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Combined rating prediction
        """
        # Get NeuMF prediction
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        if user_idx is not None and item_idx is not None:
            with torch.no_grad():
                neumf_pred = self.neumf_model.model(
                    torch.LongTensor([user_idx]).to(self.neumf_model.device),
                    torch.LongTensor([item_idx]).to(self.neumf_model.device)
                ).cpu().numpy()[0]
        else:
            neumf_pred = 3.0  # Default rating if user/item not found
            
        # Get Surprise predictions
        svd_pred = self.svd.predict(user_id, item_id).est
        
        
        # Combine predictions (weighted average)
        # NeuMF: 0.4, SVD: 0.3, SVD++: 0.3
        combined_rating = (neumf_pred * 0.4) + (svd_pred * 0.3) 
        
        return combined_rating
    
    def get_top_k_recommendations(self, user_id, k=10):
        """
        Get top k recommendations for a user
        
        Args:
            user_id: User ID
            k: Number of recommendations to return
            
        Returns:
            List of tuples (item_id, combined_rating)
        """
        # Get items not rated by the user
        rated_items = set(self.df[self.df['userId'] == user_id]['movieId'])
        all_items = set(self.df['movieId'])
        unrated_items = all_items - rated_items
        
        # Get predictions for all unrated items
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by prediction and return top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]
    
    def get_movie_titles(self):
        """Get a mapping of movie IDs to titles"""
        movies_path = os.path.join(os.path.dirname(self.data_path), 'movies.csv')
        df = pd.read_csv(movies_path)
        return dict(zip(df["movieId"], df["title"])) 
