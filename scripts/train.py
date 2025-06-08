#!/usr/bin/env python3
"""
Training script for music recommendation models.
Supports training on M1 Mac with Metal Performance Shaders (MPS) when available.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from datetime import datetime

from model import get_model

# Check for M1 Mac MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


class MusicDataset(Dataset):
    """PyTorch dataset for music recommendation data"""
    
    def __init__(self, data_path, use_features=False):
        self.data = pd.read_csv(data_path)
        self.use_features = use_features
        
        # Required columns
        self.user_ids = self.data['user_id'].values
        self.item_ids = self.data['item_id'].values
        self.ratings = self.data['rating'].values.astype(np.float32)
        
        # Optional feature columns
        if use_features:
            self.genre_ids = self.data['genre_encoded'].values
            self.language_ids = self.data['language_encoded'].values
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }
        
        if self.use_features:
            sample['genre_id'] = self.genre_ids[idx]
            sample['language_id'] = self.language_ids[idx]
            
        return sample


class Trainer:
    """Model trainer with support for different architectures"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        log_dir='runs',
        checkpoint_dir='output/models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, f'train_{timestamp}'))
        
        # Setup checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Training')
        for batch in pbar:
            # Move data to device
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if 'genre_id' in batch:
                genre_ids = batch['genre_id'].to(device)
                language_ids = batch['language_id'].to(device)
                predictions = self.model(user_ids, item_ids, genre_ids, language_ids)
            else:
                predictions = self.model(user_ids, item_ids)
                
            # Calculate loss
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / batch_count
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} - Validation')
            for batch in pbar:
                # Move data to device
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                ratings = batch['rating'].to(device)
                
                # Forward pass
                if 'genre_id' in batch:
                    genre_ids = batch['genre_id'].to(device)
                    language_ids = batch['language_id'].to(device)
                    predictions = self.model(user_ids, item_ids, genre_ids, language_ids)
                else:
                    predictions = self.model(user_ids, item_ids)
                    
                # Calculate loss
                loss = self.criterion(predictions, ratings)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return total_loss / batch_count
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_loss
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            
    def train(self, epochs, patience=10):
        """Train the model"""
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
        self.writer.close()
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train music recommendation model')
    parser.add_argument('--model_type', type=str, default='ncf', 
                        choices=['ncf', 'mf', 'hybrid'], help='Model type')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay')
    parser.add_argument('--use_features', action='store_true', 
                        help='Use content features')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    parser.add_argument('--data_dir', type=str, default='output/processed/', 
                        help='Processed data directory')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        
    print(f"Dataset info:")
    print(f"- Users: {metadata['n_users']}")
    print(f"- Items: {metadata['n_items']}")
    print(f"- Genres: {metadata['n_genres']}")
    print(f"- Languages: {metadata['n_languages']}")
    
    # Create datasets
    train_dataset = MusicDataset(
        os.path.join(args.data_dir, 'train_processed.csv'),
        use_features=args.use_features
    )
    val_dataset = MusicDataset(
        os.path.join(args.data_dir, 'val_processed.csv'),
        use_features=args.use_features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device != torch.device("cpu") else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device != torch.device("cpu") else False
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )
    
    trainer.train(epochs=args.epochs)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join('output/models', f'{args.model_type}_final.pt'))
    print(f"Model saved to output/models/{args.model_type}_final.pt")


if __name__ == '__main__':
    main()