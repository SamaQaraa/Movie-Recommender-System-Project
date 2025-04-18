class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_dim=32, layers=[64, 32, 16]):
        super(NeuMF, self).__init__()
        
        # MF embeddings
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dim)
        
        # MLP layers
        input_size = mlp_dim * 2  # Concatenated user and item embeddings
        self.mlp = nn.Sequential(
            nn.Linear(input_size, layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Add remaining layers dynamically
        for i in range(1, len(layers)):
            self.mlp.add_module(f"fc{i}", nn.Linear(layers[i-1], layers[i]))
            self.mlp.add_module(f"relu{i}", nn.ReLU())
            self.mlp.add_module(f"dropout{i}", nn.Dropout(0.2))
        
        # Final layer
        self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)
        
        # Initialize weights
        self._init_weight_()
    
    def _init_weight_(self):
        nn.init.normal_(self.mf_user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mf_item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, mean=0.0, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
    
    def forward(self, user, item):
        # MF part
        mf_user_embedded = self.mf_user_embedding(user)
        mf_item_embedded = self.mf_item_embedding(item)
        mf_output = mf_user_embedded * mf_item_embedded  # element-wise product
        
        # MLP part
        mlp_user_embedded = self.mlp_user_embedding(user)
        mlp_item_embedded = self.mlp_item_embedding(item)
        mlp_input = torch.cat([mlp_user_embedded, mlp_item_embedded], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate MF and MLP parts
        output = torch.cat([mf_output, mlp_output], dim=-1)
        output = self.predict_layer(output)
        output = torch.sigmoid(output) * 4.5 + 0.5  # Scale to [0.5, 5.0]
        
        return output.squeeze()
