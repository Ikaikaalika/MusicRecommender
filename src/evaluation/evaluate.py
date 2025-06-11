#!/usr/bin/env python3
"""
Evaluation script for music recommendation models.
Computes metrics including MRR, Hit Rate, and NDCG.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.models.pytorch.model import get_model
    from src.training.pytorch.train import MusicDataset, device
except ImportError:
    # Fallback for direct script execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'pytorch'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training', 'pytorch'))
    from model import get_model
    from train import MusicDataset, device


class Evaluator:
    """Model evaluator for recommendation metrics"""
    
    def __init__(self, model, data_loader, n_users, n_items, top_k=[5, 10, 20]):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.n_users = n_users
        self.n_items = n_items
        self.top_k = top_k
        
        # Load ground truth
        self.ground_truth = defaultdict(set)
        for batch in data_loader:
            for user_id, item_id, rating in zip(
                batch['user_id'].numpy(),
                batch['item_id'].numpy(),
                batch['rating'].numpy()
            ):
                if rating > 0.5:  # Positive interaction
                    self.ground_truth[user_id].add(item_id)
                    
    def get_user_recommendations(self, user_id, exclude_seen=True):
        """Get recommendations for a single user"""
        self.model.eval()
        
        with torch.no_grad():
            # Create tensors for all items
            user_tensor = torch.tensor([user_id] * self.n_items).to(device)
            item_tensor = torch.arange(self.n_items).to(device)
            
            # Get predictions
            predictions = self.model(user_tensor, item_tensor)
            
            # Convert to numpy
            scores = predictions.cpu().numpy()
            
            # Exclude seen items if requested
            if exclude_seen and user_id in self.ground_truth:
                seen_items = self.ground_truth[user_id]
                for item_id in seen_items:
                    scores[item_id] = -np.inf
                    
            # Get top-k items
            top_items = np.argsort(scores)[::-1]
            
            return top_items, scores
            
    def compute_mrr(self, recommendations, relevant_items):
        """Compute Mean Reciprocal Rank"""
        for i, item in enumerate(recommendations):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def compute_hit_rate(self, recommendations, relevant_items, k):
        """Compute Hit Rate @k"""
        hits = len(set(recommendations[:k]) & relevant_items)
        return 1.0 if hits > 0 else 0.0
    
    def compute_ndcg(self, recommendations, relevant_items, k):
        """Compute Normalized Discounted Cumulative Gain @k"""
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
                
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, sample_users=None):
        """Evaluate the model on all metrics"""
        print("Evaluating model...")
        
        # Sample users for evaluation if specified
        all_users = list(self.ground_truth.keys())
        if sample_users and sample_users < len(all_users):
            eval_users = np.random.choice(all_users, sample_users, replace=False)
        else:
            eval_users = all_users
            
        # Initialize metrics
        metrics = {
            'mrr': [],
            **{f'hit_rate@{k}': [] for k in self.top_k},
            **{f'ndcg@{k}': [] for k in self.top_k}
        }
        
        # Evaluate each user
        for user_id in tqdm(eval_users, desc="Evaluating users"):
            if user_id not in self.ground_truth or len(self.ground_truth[user_id]) == 0:
                continue
                
            # Get recommendations
            recommendations, _ = self.get_user_recommendations(user_id)
            relevant_items = self.ground_truth[user_id]
            
            # Compute metrics
            metrics['mrr'].append(self.compute_mrr(recommendations, relevant_items))
            
            for k in self.top_k:
                metrics[f'hit_rate@{k}'].append(
                    self.compute_hit_rate(recommendations, relevant_items, k)
                )
                metrics[f'ndcg@{k}'].append(
                    self.compute_ndcg(recommendations, relevant_items, k)
                )
                
        # Aggregate metrics
        results = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                results[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_users': len(values)
                }
            else:
                results[metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'n_users': 0
                }
                
        return results
    
    def evaluate_popularity_bias(self):
        """Evaluate popularity bias in recommendations"""
        print("Evaluating popularity bias...")
        
        # Count item frequencies in training data
        item_counts = defaultdict(int)
        for users_items in self.ground_truth.values():
            for item in users_items:
                item_counts[item] += 1
                
        # Sort items by popularity
        items_by_popularity = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Divide items into buckets
        n_buckets = 10
        bucket_size = len(items_by_popularity) // n_buckets
        item_to_bucket = {}
        
        for i, (item, _) in enumerate(items_by_popularity):
            bucket = min(i // bucket_size, n_buckets - 1)
            item_to_bucket[item] = bucket
            
        # Count recommendations per bucket
        bucket_counts = defaultdict(int)
        total_recs = 0
        
        sample_users = min(1000, len(self.ground_truth))
        eval_users = np.random.choice(list(self.ground_truth.keys()), sample_users, replace=False)
        
        for user_id in tqdm(eval_users, desc="Analyzing bias"):
            recommendations, _ = self.get_user_recommendations(user_id)
            
            for item in recommendations[:20]:  # Top-20 recommendations
                if item in item_to_bucket:
                    bucket_counts[item_to_bucket[item]] += 1
                    total_recs += 1
                    
        # Calculate distribution
        bias_distribution = {}
        for bucket in range(n_buckets):
            bias_distribution[f'bucket_{bucket}'] = {
                'percentage': (bucket_counts[bucket] / total_recs * 100) if total_recs > 0 else 0,
                'description': f'Items ranked {bucket * bucket_size + 1}-{(bucket + 1) * bucket_size}'
            }
            
        return bias_distribution


def save_results(results, output_path):
    """Save evaluation results"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate music recommendation model')
    parser.add_argument('--model_type', type=str, default='ncf', 
                        choices=['ncf', 'mf', 'hybrid'], help='Model type')
    parser.add_argument('--model_path', type=str, 
                        default='experiments/checkpoints/best_checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=2048, 
                        help='Batch size')
    parser.add_argument('--use_features', action='store_true', 
                        help='Use content features')
    parser.add_argument('--data_dir', type=str, default='experiments/processed/', 
                        help='Processed data directory')
    parser.add_argument('--sample_users', type=int, default=None,
                        help='Number of users to sample for evaluation')
    parser.add_argument('--output_file', type=str, default='experiments/evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        
    # Create dataset
    val_dataset = MusicDataset(
        os.path.join(args.data_dir, 'val_processed.csv'),
        use_features=args.use_features
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
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
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data_loader=val_loader,
        n_users=metadata['n_users'],
        n_items=metadata['n_items']
    )
    
    # Evaluate
    results = {
        'model_type': args.model_type,
        'metrics': evaluator.evaluate(sample_users=args.sample_users),
        'popularity_bias': evaluator.evaluate_popularity_bias()
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nRanking Metrics:")
    for metric_name, values in results['metrics'].items():
        print(f"{metric_name:20s}: {values['mean']:.4f} ± {values['std']:.4f}")
        
    print("\nPopularity Bias Analysis:")
    for bucket, info in results['popularity_bias'].items():
        print(f"{info['description']:40s}: {info['percentage']:.2f}%")
        
    # Save results
    save_results(results, args.output_file)


if __name__ == '__main__':
    main()