#!/usr/bin/env python3
"""
Neural Collaborative Filtering model for music recommendation.
Implements matrix factorization with neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class NCF(nn.Module):
    """Neural Collaborative Filtering model"""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        hidden_layers: list = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_features: bool = False,
        n_genres: int = 0,
        n_languages: int = 0
    ):
        super(NCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_features = use_features
        
        # User and item embeddings for GMF (Generalized Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # User and item embeddings for MLP
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        # Additional feature embeddings if using metadata
        if use_features:
            self.genre_embedding = nn.Embedding(n_genres, 32)
            self.language_embedding = nn.Embedding(n_languages, 16)
            mlp_input_dim = 2 * embedding_dim + 32 + 16
        else:
            mlp_input_dim = 2 * embedding_dim
            
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = mlp_input_dim
        
        for hidden_dim in hidden_layers:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        # Final prediction layer
        self.prediction = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_ids: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        
        # GMF part
        user_embedding_gmf = self.user_embedding_gmf(user_ids)
        item_embedding_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_embedding_gmf * item_embedding_gmf
        
        # MLP part
        user_embedding_mlp = self.user_embedding_mlp(user_ids)
        item_embedding_mlp = self.item_embedding_mlp(item_ids)
        
        if self.use_features and genre_ids is not None and language_ids is not None:
            genre_embedding = self.genre_embedding(genre_ids)
            language_embedding = self.language_embedding(language_ids)
            mlp_input = torch.cat([
                user_embedding_mlp,
                item_embedding_mlp,
                genre_embedding,
                language_embedding
            ], dim=1)
        else:
            mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=1)
            
        # Pass through MLP layers
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
            
        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        prediction = self.prediction(concat_output)
        
        return torch.sigmoid(prediction).squeeze()
    
    def get_user_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user embeddings for both GMF and MLP"""
        return self.user_embedding_gmf.weight, self.user_embedding_mlp.weight
    
    def get_item_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item embeddings for both GMF and MLP"""
        return self.item_embedding_gmf.weight, self.item_embedding_mlp.weight


class MatrixFactorization(nn.Module):
    """Simple Matrix Factorization model as baseline"""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        use_bias: bool = True
    ):
        super(MatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Biases
        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
            
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        prediction = (user_emb * item_emb).sum(dim=1)
        
        # Add biases
        if self.use_bias:
            prediction += self.user_bias(user_ids).squeeze()
            prediction += self.item_bias(item_ids).squeeze()
            prediction += self.global_bias
            
        return torch.sigmoid(prediction)
    
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings"""
        return self.user_embedding.weight, self.item_embedding.weight


class HybridModel(nn.Module):
    """Hybrid model combining collaborative filtering with content features"""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        n_languages: int,
        embedding_dim: int = 128,
        content_dim: int = 64,
        hidden_dims: list = [256, 128, 64]
    ):
        super(HybridModel, self).__init__()
        
        # Collaborative embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Content embeddings
        self.genre_embedding = nn.Embedding(n_genres, content_dim // 2)
        self.language_embedding = nn.Embedding(n_languages, content_dim // 4)
        
        # Combine features
        total_dim = 2 * embedding_dim + content_dim // 2 + content_dim // 4
        
        # MLP
        layers = []
        input_dim = total_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.genre_embedding.weight)
        nn.init.xavier_uniform_(self.language_embedding.weight)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        language_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genre_ids)
        lang_emb = self.language_embedding(language_ids)
        
        # Concatenate all features
        features = torch.cat([user_emb, item_emb, genre_emb, lang_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(features)
        
        return torch.sigmoid(output).squeeze()


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to get model by type"""
    if model_type == 'ncf':
        return NCF(**kwargs)
    elif model_type == 'mf':
        return MatrixFactorization(**kwargs)
    elif model_type == 'hybrid':
        return HybridModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")