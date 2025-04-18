import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .inference import NeuMF

def train_model(data_path, save_path='recommender_model/'):
    """
    Train the NeuMF model and save it
    
    Args:
        data_path: Path to the ratings.csv file
        save_path: Directory to save the trained model
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Create mappings
    user_ids = df["userId"].unique()
    item_ids = df["movieId"].unique()
    
    # Normalize ratings to [0, 1]
    df["rating"] = (df["rating"] - 0.5) / 4.5
    
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    
    # Convert to indices
    df["user_idx"] = df["userId"].map(user_to_idx)
    df["item_idx"] = df["movieId"].map(item_to_idx)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create PyTorch Dataset
    class RatingDataset(Dataset):
        def __init__(self, df):
            self.users = torch.LongTensor(df["user_idx"].values)
            self.items = torch.LongTensor(df["item_idx"].values)
            self.ratings = torch.FloatTensor(df["rating"].values)
            
        def __len__(self):
            return len(self.users)
        
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx], self.ratings[idx]
    
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)
    
    # Initialize model
    num_users = len(user_ids)
    num_items = len(item_ids)
    model = NeuMF(num_users, num_items, mf_dim=16, mlp_dim=64, layers=[128, 64, 32])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (users, items, ratings) in enumerate(dataloader):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    # Training
    num_epochs = 5
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
        
        # Save the best model
        if train_loss < best_loss:
            best_loss = train_loss
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict.pth'))
            with open(os.path.join(save_path, 'mappings.pkl'), 'wb') as f:
                import pickle
                pickle.dump({
                    'user_to_idx': user_to_idx,
                    'item_to_idx': item_to_idx,
                    'num_users': num_users,
                    'num_items': num_items
                }, f)
    
    return model, user_to_idx, item_to_idx

if __name__ == "__main__":
    # Use the absolute workspace path
    workspace_path = "D:/ITI/Rec_Sys_Intake_45/project_descrption/project"
    data_path = os.path.join(workspace_path, 'data', 'ml-latest-small', 'ratings.csv')
    
    # Create the recommender_model directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'recommender_model')
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    print(f"Saving model to: {save_path}")
    
    # Train the model
    model, user_to_idx, item_to_idx = train_model(data_path, save_path)

