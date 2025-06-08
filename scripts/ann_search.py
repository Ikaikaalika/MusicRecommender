#!/usr/bin/env python3
"""
Approximate Nearest Neighbor (ANN) search using FAISS for fast recommendation retrieval.
Implements embedding-based similarity search with SBERT-style embeddings.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
from typing import List, Tuple, Dict

from model import get_model
from train import device


class EmbeddingRetriever:
    """Fast embedding-based retrieval using FAISS"""
    
    def __init__(
        self,
        model,
        n_users: int,
        n_items: int,
        embedding_type: str = 'collaborative',
        use_gpu: bool = False
    ):
        self.model = model.to(device)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_type = embedding_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize FAISS index
        self.index = None
        self.item_ids = None
        
    def extract_embeddings(self):
        """Extract embeddings from the trained model"""
        print("Extracting embeddings from model...")
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'get_embeddings'):
                # For MF model
                user_emb, item_emb = self.model.get_embeddings()
            elif hasattr(self.model, 'get_item_embeddings'):
                # For NCF model
                item_emb_gmf, item_emb_mlp = self.model.get_item_embeddings()
                # Concatenate GMF and MLP embeddings
                item_emb = torch.cat([item_emb_gmf, item_emb_mlp], dim=1)
                
                user_emb_gmf, user_emb_mlp = self.model.get_user_embeddings()
                user_emb = torch.cat([user_emb_gmf, user_emb_mlp], dim=1)
            else:
                # For hybrid model
                item_emb = self.model.item_embedding.weight
                user_emb = self.model.user_embedding.weight
                
        # Convert to numpy
        self.user_embeddings = user_emb.cpu().numpy()
        self.item_embeddings = item_emb.cpu().numpy()
        
        print(f"User embeddings shape: {self.user_embeddings.shape}")
        print(f"Item embeddings shape: {self.item_embeddings.shape}")
        
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        if self.use_gpu:
            # Use GPU index
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            # Use CPU index with IVF for large datasets
            if embeddings.shape[0] > 50000:
                # Use IVF index for faster search on large datasets
                nlist = int(np.sqrt(embeddings.shape[0]))
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(embeddings)
            else:
                # Use flat index for smaller datasets
                self.index = faiss.IndexFlatIP(dimension)
                
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        
    def search_similar_items(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar items given a query embedding"""
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        return indices[0], distances[0]
    
    def get_user_recommendations(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: set = None
    ) -> List[Tuple[int, float]]:
        """Get recommendations for a user"""
        if user_id >= self.n_users:
            raise ValueError(f"User ID {user_id} out of range")
            
        # Get user embedding
        user_embedding = self.user_embeddings[user_id]
        
        # Search for similar items
        if self.embedding_type == 'collaborative':
            # Use item embeddings directly
            indices, scores = self.search_similar_items(user_embedding, k * 3)  # Get more to filter
        else:
            # For hybrid approach, combine collaborative and content signals
            indices, scores = self.search_similar_items(user_embedding, k * 3)
            
        # Filter out seen items if requested
        recommendations = []
        for idx, score in zip(indices, scores):
            if exclude_seen and seen_items and idx in seen_items:
                continue
            recommendations.append((idx, float(score)))
            if len(recommendations) >= k:
                break
                
        return recommendations
    
    def get_item_to_item_recommendations(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get similar items for a given item"""
        if item_id >= self.n_items:
            raise ValueError(f"Item ID {item_id} out of range")
            
        # Get item embedding
        item_embedding = self.item_embeddings[item_id]
        
        # Search for similar items (k+1 to exclude the item itself)
        indices, scores = self.search_similar_items(item_embedding, k + 1)
        
        # Exclude the item itself
        recommendations = [(idx, float(score)) for idx, score in zip(indices, scores) if idx != item_id]
        
        return recommendations[:k]
    
    def save_index(self, output_dir: str):
        """Save FAISS index and embeddings"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        
        # Save embeddings
        np.save(os.path.join(output_dir, 'user_embeddings.npy'), self.user_embeddings)
        np.save(os.path.join(output_dir, 'item_embeddings.npy'), self.item_embeddings)
        
        print(f"Index and embeddings saved to {output_dir}")
        
    def load_index(self, output_dir: str):
        """Load FAISS index and embeddings"""
        # Load FAISS index
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)
        
        # Load embeddings
        self.user_embeddings = np.load(os.path.join(output_dir, 'user_embeddings.npy'))
        self.item_embeddings = np.load(os.path.join(output_dir, 'item_embeddings.npy'))
        
        print(f"Index and embeddings loaded from {output_dir}")


class HybridRetriever(EmbeddingRetriever):
    """Hybrid retriever combining collaborative filtering and content-based features"""
    
    def __init__(self, model, n_users, n_items, songs_df=None, use_sbert=False):
        super().__init__(model, n_users, n_items, 'hybrid')
        self.songs_df = songs_df
        self.use_sbert = use_sbert
        
        if use_sbert:
            print("Loading sentence transformer model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
    def create_content_embeddings(self):
        """Create content-based embeddings from song metadata"""
        if self.songs_df is None:
            raise ValueError("Songs dataframe required for content embeddings")
            
        print("Creating content embeddings...")
        
        if self.use_sbert:
            # Create text descriptions for songs
            descriptions = []
            for _, song in tqdm(self.songs_df.iterrows(), desc="Creating descriptions"):
                desc = f"{song.get('artist_name', '')} {song.get('composer', '')} {song.get('lyricist', '')}"
                descriptions.append(desc)
                
            # Encode with SBERT
            self.content_embeddings = self.sbert_model.encode(
                descriptions,
                batch_size=256,
                show_progress_bar=True
            )
        else:
            # Use genre and language as simple one-hot encodings
            # This is a simplified approach - in production, you'd use more sophisticated encoding
            genre_vocab = set()
            for genres in self.songs_df['genre_ids'].dropna():
                genre_vocab.update(str(genres).split('|'))
            
            genre_to_idx = {g: i for i, g in enumerate(sorted(genre_vocab))}
            
            # Create content vectors
            content_vectors = []
            for _, song in self.songs_df.iterrows():
                vector = np.zeros(len(genre_vocab) + 10)  # +10 for language encoding
                
                # Encode genres
                if pd.notna(song['genre_ids']):
                    for genre in str(song['genre_ids']).split('|'):
                        if genre in genre_to_idx:
                            vector[genre_to_idx[genre]] = 1
                            
                # Simple language encoding
                if pd.notna(song['language']):
                    lang_idx = int(song['language']) % 10
                    vector[len(genre_vocab) + lang_idx] = 1
                    
                content_vectors.append(vector)
                
            self.content_embeddings = np.array(content_vectors)
            
    def create_hybrid_embeddings(self, alpha=0.7):
        """Combine collaborative and content embeddings"""
        print("Creating hybrid embeddings...")
        
        # Normalize embeddings
        collab_norm = self.item_embeddings / np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        content_norm = self.content_embeddings / np.linalg.norm(self.content_embeddings, axis=1, keepdims=True)
        
        # Combine with weighting
        self.hybrid_embeddings = alpha * collab_norm + (1 - alpha) * content_norm
        
        return self.hybrid_embeddings


def generate_recommendations_batch(retriever, user_ids, k=10, output_file=None):
    """Generate recommendations for multiple users"""
    print(f"Generating recommendations for {len(user_ids)} users...")
    
    recommendations = {}
    
    for user_id in tqdm(user_ids):
        try:
            recs = retriever.get_user_recommendations(user_id, k=k)
            recommendations[user_id] = [
                {'item_id': int(item_id), 'score': float(score)}
                for item_id, score in recs
            ]
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            recommendations[user_id] = []
            
    if output_file:
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Recommendations saved to {output_file}")
        
        # Also save as CSV for easy analysis
        csv_data = []
        for user_id, recs in recommendations.items():
            for rank, rec in enumerate(recs):
                csv_data.append({
                    'user_id': user_id,
                    'item_id': rec['item_id'],
                    'score': rec['score'],
                    'rank': rank + 1
                })
        
        csv_file = output_file.replace('.json', '.csv')
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"CSV version saved to {csv_file}")
        
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Generate recommendations using ANN search')
    parser.add_argument('--model_type', type=str, default='ncf', 
                        choices=['ncf', 'mf', 'hybrid'], help='Model type')
    parser.add_argument('--model_path', type=str, 
                        default='output/models/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Embedding dimension')
    parser.add_argument('--use_features', action='store_true', 
                        help='Use content features')
    parser.add_argument('--data_dir', type=str, default='output/processed/', 
                        help='Processed data directory')
    parser.add_argument('--index_dir', type=str, default='output/embeddings/',
                        help='Directory to save/load FAISS index')
    parser.add_argument('--user_id', type=int, default=None,
                        help='Generate recommendations for specific user')
    parser.add_argument('--n_users', type=int, default=100,
                        help='Number of users to generate recommendations for')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recommendations per user')
    parser.add_argument('--output_file', type=str, default='output/recommendations.json',
                        help='Output file for recommendations')
    parser.add_argument('--build_index', action='store_true',
                        help='Build new FAISS index')
    parser.add_argument('--use_hybrid', action='store_true',
                        help='Use hybrid retrieval')
    parser.add_argument('--use_sbert', action='store_true',
                        help='Use SBERT for content embeddings')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        
    # Create model
    model_kwargs = {
        'n_users': metadata['n_users'],
        'n_items': metadata['n_items'],
        'embedding_dim': args.embedding_dim
    }
    
    if args.use_features:
        model_kwargs['use_features'] = True
        model_kwargs['n_genres'] = metadata['n_genres']
        model_kwargs['n_languages'] = metadata['n_languages']
        
    model = get_model(args.model_type, **model_kwargs)
    
    # Load model weights
    if args.model_path.endswith('.pt'):
        if 'checkpoint' in args.model_path:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
    # Create retriever
    if args.use_hybrid:
        # Load songs data for hybrid approach
        songs_df = pd.read_csv(os.path.join(args.data_dir.replace('processed', ''), 'songs.csv'))
        retriever = HybridRetriever(
            model,
            metadata['n_users'],
            metadata['n_items'],
            songs_df,
            use_sbert=args.use_sbert
        )
    else:
        retriever = EmbeddingRetriever(
            model,
            metadata['n_users'],
            metadata['n_items']
        )
        
    # Build or load index
    if args.build_index or not os.path.exists(os.path.join(args.index_dir, 'faiss_index.bin')):
        # Extract embeddings
        retriever.extract_embeddings()
        
        # Create content embeddings for hybrid
        if args.use_hybrid:
            retriever.create_content_embeddings()
            retriever.create_hybrid_embeddings()
            embeddings_to_index = retriever.hybrid_embeddings
        else:
            embeddings_to_index = retriever.item_embeddings
            
        # Build index
        retriever.build_index(embeddings_to_index)
        
        # Save index
        retriever.save_index(args.index_dir)
    else:
        # Load existing index
        retriever.load_index(args.index_dir)
        
    # Generate recommendations
    if args.user_id is not None:
        # Single user
        print(f"\nGenerating recommendations for user {args.user_id}...")
        recommendations = retriever.get_user_recommendations(args.user_id, k=args.top_k)
        
        print(f"\nTop {args.top_k} recommendations:")
        print("-" * 50)
        for rank, (item_id, score) in enumerate(recommendations, 1):
            print(f"{rank}. Item {item_id}: score = {score:.4f}")
            
    else:
        # Batch generation
        # Sample random users
        all_user_ids = list(range(metadata['n_users']))
        sample_user_ids = np.random.choice(all_user_ids, min(args.n_users, len(all_user_ids)), replace=False)
        
        recommendations = generate_recommendations_batch(
            retriever,
            sample_user_ids,
            k=args.top_k,
            output_file=args.output_file
        )
        
        print(f"\nGenerated recommendations for {len(recommendations)} users")
        
    # Demonstrate item-to-item recommendations
    if not args.user_id:
        print("\n" + "="*50)
        print("ITEM-TO-ITEM RECOMMENDATIONS DEMO")
        print("="*50)
        
        sample_item = np.random.randint(0, metadata['n_items'])
        similar_items = retriever.get_item_to_item_recommendations(sample_item, k=5)
        
        print(f"\nItems similar to item {sample_item}:")
        for rank, (item_id, score) in enumerate(similar_items, 1):
            print(f"{rank}. Item {item_id}: similarity = {score:.4f}")


if __name__ == '__main__':
    main()