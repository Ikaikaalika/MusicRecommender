#!/usr/bin/env python3
"""
Data preprocessing script for KKBOX music recommendation dataset.
Handles data cleaning, feature engineering, and train/validation split.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
        
        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()
        
    def load_data(self):
        """Load raw data files"""
        print("Loading data files...")
        
        # Load training data
        train_path = os.path.join(self.input_dir, 'train.csv')
        self.train_df = pd.read_csv(train_path)
        print(f"Loaded train.csv: {self.train_df.shape}")
        
        # Load songs metadata
        songs_path = os.path.join(self.input_dir, 'songs.csv')
        self.songs_df = pd.read_csv(songs_path)
        print(f"Loaded songs.csv: {self.songs_df.shape}")
        
        # Load members data if available
        members_path = os.path.join(self.input_dir, 'members.csv')
        if os.path.exists(members_path):
            self.members_df = pd.read_csv(members_path)
            print(f"Loaded members.csv: {self.members_df.shape}")
        else:
            self.members_df = None
            
    def clean_data(self):
        """Clean and filter the data"""
        print("\nCleaning data...")
        
        # Remove duplicates
        original_size = len(self.train_df)
        self.train_df = self.train_df.drop_duplicates(['msno', 'song_id'])
        print(f"Removed {original_size - len(self.train_df)} duplicate interactions")
        
        # Filter out users and songs with very few interactions
        min_user_interactions = 5
        min_song_interactions = 5
        
        user_counts = self.train_df['msno'].value_counts()
        song_counts = self.train_df['song_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_songs = song_counts[song_counts >= min_song_interactions].index
        
        self.train_df = self.train_df[
            self.train_df['msno'].isin(valid_users) & 
            self.train_df['song_id'].isin(valid_songs)
        ]
        
        print(f"Filtered data shape: {self.train_df.shape}")
        print(f"Unique users: {self.train_df['msno'].nunique()}")
        print(f"Unique songs: {self.train_df['song_id'].nunique()}")
        
    def encode_ids(self):
        """Encode user and song IDs to continuous integers"""
        print("\nEncoding IDs...")
        
        # Encode user IDs
        self.train_df['user_id'] = self.user_encoder.fit_transform(self.train_df['msno'])
        
        # Encode song IDs
        self.train_df['item_id'] = self.song_encoder.fit_transform(self.train_df['song_id'])
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.song_encoder.classes_)
        
        print(f"Number of users: {self.n_users}")
        print(f"Number of items: {self.n_items}")
        
    def create_implicit_feedback(self):
        """Convert explicit ratings to implicit feedback"""
        print("\nCreating implicit feedback...")
        
        # For KKBOX, target=1 means the user listened to the song again within a month
        # We'll use this as implicit positive feedback
        self.train_df['rating'] = self.train_df['target'].astype(float)
        
        # Add confidence weights based on source system tab
        source_weights = {
            'my library': 1.5,
            'search': 1.3,
            'discover': 1.2,
            'radio': 1.0,
            'explore': 1.1
        }
        
        if 'source_system_tab' in self.train_df.columns:
            self.train_df['confidence'] = self.train_df['source_system_tab'].map(
                lambda x: source_weights.get(x, 1.0)
            )
        else:
            self.train_df['confidence'] = 1.0
            
    def create_features(self):
        """Create additional features from songs metadata"""
        print("\nCreating features...")
        
        # Merge with songs metadata
        self.train_df = self.train_df.merge(
            self.songs_df[['song_id', 'song_length', 'genre_ids', 'artist_name', 'language']],
            on='song_id',
            how='left'
        )
        
        # Process song length (convert to minutes)
        self.train_df['song_length_min'] = self.train_df['song_length'] / 60000.0
        
        # Extract primary genre
        self.train_df['primary_genre'] = self.train_df['genre_ids'].apply(
            lambda x: str(x).split('|')[0] if pd.notna(x) else '-1'
        )
        
        # Encode categorical features
        le_genre = LabelEncoder()
        le_language = LabelEncoder()
        
        self.train_df['genre_encoded'] = le_genre.fit_transform(
            self.train_df['primary_genre'].fillna('-1')
        )
        self.train_df['language_encoded'] = le_language.fit_transform(
            self.train_df['language'].fillna(-1)
        )
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and validation sets"""
        print("\nSplitting data...")
        
        # Temporal split would be more realistic, but for simplicity we'll use random split
        train_data, val_data = train_test_split(
            self.train_df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.train_df['rating']
        )
        
        print(f"Training set: {len(train_data)} interactions")
        print(f"Validation set: {len(val_data)} interactions")
        
        return train_data, val_data
        
    def save_processed_data(self, train_data, val_data):
        """Save processed data and encoders"""
        print("\nSaving processed data...")
        
        # Save train/val data
        train_data.to_csv(
            os.path.join(self.output_dir, 'processed', 'train_processed.csv'),
            index=False
        )
        val_data.to_csv(
            os.path.join(self.output_dir, 'processed', 'val_processed.csv'),
            index=False
        )
        
        # Save encoders
        with open(os.path.join(self.output_dir, 'processed', 'user_encoder.pkl'), 'wb') as f:
            pickle.dump(self.user_encoder, f)
        with open(os.path.join(self.output_dir, 'processed', 'song_encoder.pkl'), 'wb') as f:
            pickle.dump(self.song_encoder, f)
            
        # Save metadata
        metadata = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_genres': len(self.train_df['genre_encoded'].unique()),
            'n_languages': len(self.train_df['language_encoded'].unique())
        }
        with open(os.path.join(self.output_dir, 'processed', 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        print("Processing complete!")
        
    def create_interaction_matrix(self, data):
        """Create sparse interaction matrix for collaborative filtering"""
        from scipy.sparse import csr_matrix
        
        # Create sparse matrix
        row = data['user_id'].values
        col = data['item_id'].values
        data_values = data['rating'].values * data['confidence'].values
        
        interaction_matrix = csr_matrix(
            (data_values, (row, col)),
            shape=(self.n_users, self.n_items)
        )
        
        return interaction_matrix
        
    def run(self):
        """Run the complete preprocessing pipeline"""
        self.load_data()
        self.clean_data()
        self.encode_ids()
        self.create_implicit_feedback()
        self.create_features()
        
        train_data, val_data = self.split_data()
        self.save_processed_data(train_data, val_data)
        
        # Create and save interaction matrices
        train_matrix = self.create_interaction_matrix(train_data)
        val_matrix = self.create_interaction_matrix(val_data)
        
        from scipy.sparse import save_npz
        save_npz(os.path.join(self.output_dir, 'processed', 'train_matrix.npz'), train_matrix)
        save_npz(os.path.join(self.output_dir, 'processed', 'val_matrix.npz'), val_matrix)
        
        return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description='Preprocess KKBOX music recommendation data')
    parser.add_argument('--input_dir', type=str, default='data/', help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/', help='Output directory')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.input_dir, args.output_dir)
    preprocessor.run()

if __name__ == '__main__':
    main()