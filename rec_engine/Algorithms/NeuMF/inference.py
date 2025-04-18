import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_dim=32, layers=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_dim
        
        # Matrix Factorization embeddings
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dim)
        
        # MLP layers
        self.mlp = nn.Sequential()
        input_size = mlp_dim * 2
        for i, layer_size in enumerate(layers):
            self.mlp.add_module(f'layer_{i}', nn.Linear(input_size, layer_size))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            self.mlp.add_module(f'dropout_{i}', nn.Dropout(0.2))
            input_size = layer_size
        
        # Final prediction layer
        self.prediction = nn.Linear(mf_dim + layers[-1], 1)
        
        # Initialize weights
        self._init_weight_()
    
    def _init_weight_(self):
        # Initialize embeddings
        nn.init.normal_(self.mf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        # Initialize MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize prediction layer
        nn.init.kaiming_uniform_(self.prediction.weight, a=1)
        nn.init.zeros_(self.prediction.bias)
    
    def forward(self, user, item):
        # Matrix Factorization path
        mf_user_latent = self.mf_user_embedding(user)
        mf_item_latent = self.mf_item_embedding(item)
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)
        
        # MLP path
        mlp_user_latent = self.mlp_user_embedding(user)
        mlp_item_latent = self.mlp_item_embedding(item)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        # Concatenate MF and MLP paths
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        # Final prediction
        prediction = self.prediction(vector)
        prediction = torch.sigmoid(prediction) * 4.5 + 0.5  # Scale to [0.5, 5.0]
        
        return prediction
