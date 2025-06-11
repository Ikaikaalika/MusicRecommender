#!/usr/bin/env python3
"""
Music Recommender GUI
A Tkinter-based graphical interface to demonstrate ANN search functionality
for both PyTorch and MLX music recommendation models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our recommendation modules
try:
    from src.search.ann_search import EmbeddingRetriever as PyTorchRetriever
    from src.search.ann_search_mlx import MLXEmbeddingRetriever
    from src.models.pytorch.model import get_model as get_pytorch_model
    from src.models.mlx.model_mlx import get_model as get_mlx_model
    import torch
    import mlx.core as mx
    PYTORCH_AVAILABLE = True
    MLX_AVAILABLE = True
except ImportError as e:
    print(f"Import warning: {e}")
    PYTORCH_AVAILABLE = False
    MLX_AVAILABLE = False


class MusicRecommenderGUI:
    """Main GUI application for music recommendations"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Music Recommender - ANN Search Demo")
        self.root.geometry("900x700")
        
        # Application state
        self.pytorch_retriever = None
        self.mlx_retriever = None
        self.metadata = None
        self.current_model_type = "pytorch"
        self.is_loading = False
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
        # Load default data if available
        self.load_default_data()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Tab 1: Model Selection and Loading
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="Model Setup")
        
        # Tab 2: User Recommendations
        self.user_rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.user_rec_frame, text="User Recommendations")
        
        # Tab 3: Item-to-Item Similarity
        self.item_sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.item_sim_frame, text="Item Similarity")
        
        # Tab 4: Batch Processing
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Processing")
        
        self.create_model_tab()
        self.create_user_rec_tab()
        self.create_item_sim_tab()
        self.create_batch_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a model to begin")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        
    def create_model_tab(self):
        """Create model selection and loading tab"""
        
        # Model type selection
        model_type_frame = ttk.LabelFrame(self.model_frame, text="Model Type", padding=10)
        
        self.model_type_var = tk.StringVar(value="pytorch")
        ttk.Radiobutton(model_type_frame, text="PyTorch Model", 
                       variable=self.model_type_var, value="pytorch",
                       command=self.on_model_type_change).pack(anchor='w')
        ttk.Radiobutton(model_type_frame, text="MLX Model", 
                       variable=self.model_type_var, value="mlx",
                       command=self.on_model_type_change).pack(anchor='w')
        
        # Model file selection
        file_frame = ttk.LabelFrame(self.model_frame, text="Model Files", padding=10)
        
        # PyTorch model path
        ttk.Label(file_frame, text="PyTorch Model:").grid(row=0, column=0, sticky='w', padx=5)
        self.pytorch_model_var = tk.StringVar(value="experiments/checkpoints/best_checkpoint.pt")
        ttk.Entry(file_frame, textvariable=self.pytorch_model_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", 
                  command=lambda: self.browse_file(self.pytorch_model_var, "PyTorch Model", "*.pt")).grid(row=0, column=2, padx=5)
        
        # MLX model path
        ttk.Label(file_frame, text="MLX Model:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.mlx_model_var = tk.StringVar(value="experiments/checkpoints/ncf_mlx.safetensors")
        ttk.Entry(file_frame, textvariable=self.mlx_model_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", 
                  command=lambda: self.browse_file(self.mlx_model_var, "MLX Model", "*.safetensors")).grid(row=1, column=2, padx=5, pady=5)
        
        # Data directory
        ttk.Label(file_frame, text="Data Directory:").grid(row=2, column=0, sticky='w', padx=5)
        self.data_dir_var = tk.StringVar(value="experiments/processed/")
        ttk.Entry(file_frame, textvariable=self.data_dir_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.data_dir_var)).grid(row=2, column=2, padx=5)
        
        # Model configuration
        config_frame = ttk.LabelFrame(self.model_frame, text="Model Configuration", padding=10)
        
        ttk.Label(config_frame, text="Model Architecture:").grid(row=0, column=0, sticky='w', padx=5)
        self.arch_var = tk.StringVar(value="ncf")
        arch_combo = ttk.Combobox(config_frame, textvariable=self.arch_var, 
                                 values=["ncf", "mf", "hybrid"], state="readonly")
        arch_combo.grid(row=0, column=1, padx=5, sticky='w')
        
        ttk.Label(config_frame, text="Embedding Dimension:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.embedding_dim_var = tk.StringVar(value="128")
        ttk.Entry(config_frame, textvariable=self.embedding_dim_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Load model button
        self.load_button = ttk.Button(self.model_frame, text="Load Model", command=self.load_model)
        
        # Model status
        self.model_status_text = scrolledtext.ScrolledText(self.model_frame, height=8, width=80)
        
        # Pack model tab widgets
        model_type_frame.pack(fill='x', padx=10, pady=5)
        file_frame.pack(fill='x', padx=10, pady=5)
        config_frame.pack(fill='x', padx=10, pady=5)
        self.load_button.pack(pady=10)
        ttk.Label(self.model_frame, text="Model Status:").pack(anchor='w', padx=10)
        self.model_status_text.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_user_rec_tab(self):
        """Create user recommendations tab"""
        
        # Input frame
        input_frame = ttk.LabelFrame(self.user_rec_frame, text="User Input", padding=10)
        
        ttk.Label(input_frame, text="User ID:").grid(row=0, column=0, sticky='w', padx=5)
        self.user_id_var = tk.StringVar(value="42")
        ttk.Entry(input_frame, textvariable=self.user_id_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Number of Recommendations:").grid(row=0, column=2, sticky='w', padx=15)
        self.user_top_k_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.user_top_k_var, width=10).grid(row=0, column=3, padx=5)
        
        self.get_user_recs_button = ttk.Button(input_frame, text="Get Recommendations", 
                                             command=self.get_user_recommendations)
        self.get_user_recs_button.grid(row=0, column=4, padx=15)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.user_rec_frame, text="Recommendations", padding=10)
        
        # Treeview for recommendations
        columns = ("Rank", "Item ID", "Score")
        self.user_rec_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.user_rec_tree.heading(col, text=col)
            self.user_rec_tree.column(col, width=150, anchor='center')
            
        scrollbar1 = ttk.Scrollbar(results_frame, orient='vertical', command=self.user_rec_tree.yview)
        self.user_rec_tree.configure(yscrollcommand=scrollbar1.set)
        
        # Pack user rec tab widgets
        input_frame.pack(fill='x', padx=10, pady=5)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.user_rec_tree.pack(side='left', fill='both', expand=True)
        scrollbar1.pack(side='right', fill='y')
        
    def create_item_sim_tab(self):
        """Create item similarity tab"""
        
        # Input frame
        input_frame = ttk.LabelFrame(self.item_sim_frame, text="Item Input", padding=10)
        
        ttk.Label(input_frame, text="Item ID:").grid(row=0, column=0, sticky='w', padx=5)
        self.item_id_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.item_id_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Number of Similar Items:").grid(row=0, column=2, sticky='w', padx=15)
        self.item_top_k_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.item_top_k_var, width=10).grid(row=0, column=3, padx=5)
        
        self.get_item_sim_button = ttk.Button(input_frame, text="Find Similar Items", 
                                            command=self.get_item_similarity)
        self.get_item_sim_button.grid(row=0, column=4, padx=15)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.item_sim_frame, text="Similar Items", padding=10)
        
        # Treeview for similar items
        columns = ("Rank", "Item ID", "Similarity")
        self.item_sim_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.item_sim_tree.heading(col, text=col)
            self.item_sim_tree.column(col, width=150, anchor='center')
            
        scrollbar2 = ttk.Scrollbar(results_frame, orient='vertical', command=self.item_sim_tree.yview)
        self.item_sim_tree.configure(yscrollcommand=scrollbar2.set)
        
        # Pack item sim tab widgets
        input_frame.pack(fill='x', padx=10, pady=5)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.item_sim_tree.pack(side='left', fill='both', expand=True)
        scrollbar2.pack(side='right', fill='y')
        
    def create_batch_tab(self):
        """Create batch processing tab"""
        
        # Input frame
        input_frame = ttk.LabelFrame(self.batch_frame, text="Batch Parameters", padding=10)
        
        ttk.Label(input_frame, text="Number of Users:").grid(row=0, column=0, sticky='w', padx=5)
        self.batch_users_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.batch_users_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Recommendations per User:").grid(row=0, column=2, sticky='w', padx=15)
        self.batch_top_k_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.batch_top_k_var, width=10).grid(row=0, column=3, padx=5)
        
        self.batch_process_button = ttk.Button(input_frame, text="Generate Batch Recommendations", 
                                             command=self.batch_process)
        self.batch_process_button.grid(row=0, column=4, padx=15)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.batch_frame, text="Progress", padding=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=400)
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        
        # Results frame
        results_frame = ttk.LabelFrame(self.batch_frame, text="Batch Results", padding=10)
        
        self.batch_results_text = scrolledtext.ScrolledText(results_frame, height=12, width=80)
        
        # Pack batch tab widgets
        input_frame.pack(fill='x', padx=10, pady=5)
        progress_frame.pack(fill='x', padx=10, pady=5)
        self.progress_bar.pack(pady=5)
        self.progress_label.pack()
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.batch_results_text.pack(fill='both', expand=True)
        
    def setup_layout(self):
        """Setup the main layout"""
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.status_bar.pack(side='bottom', fill='x')
        
    def browse_file(self, var, title, filetypes):
        """Browse for a file"""
        filename = filedialog.askopenfilename(
            title=f"Select {title}",
            filetypes=[(title, filetypes), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
            
    def browse_directory(self, var):
        """Browse for a directory"""
        dirname = filedialog.askdirectory(title="Select Data Directory")
        if dirname:
            var.set(dirname)
            
    def on_model_type_change(self):
        """Handle model type change"""
        self.current_model_type = self.model_type_var.get()
        self.update_status(f"Model type changed to {self.current_model_type}")
        
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def log_message(self, message):
        """Log message to model status text"""
        self.model_status_text.insert(tk.END, f"{message}\n")
        self.model_status_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_default_data(self):
        """Try to load default data if available"""
        try:
            data_dir = "experiments/processed/"
            if os.path.exists(data_dir):
                self.data_dir_var.set(data_dir)
                
            # Check for model files
            pytorch_model = "experiments/checkpoints/best_checkpoint.pt"
            if os.path.exists(pytorch_model):
                self.pytorch_model_var.set(pytorch_model)
                
            mlx_model = "experiments/checkpoints/ncf_mlx.safetensors"
            if os.path.exists(mlx_model):
                self.mlx_model_var.set(mlx_model)
                
            self.log_message("Default paths loaded successfully")
        except Exception as e:
            self.log_message(f"Could not load default paths: {e}")
            
    def load_model(self):
        """Load the selected model in a separate thread"""
        if self.is_loading:
            messagebox.showwarning("Warning", "Model is already loading...")
            return
            
        # Disable load button
        self.load_button.config(state='disabled')
        self.is_loading = True
        
        # Start loading in separate thread
        thread = threading.Thread(target=self._load_model_thread)
        thread.daemon = True
        thread.start()
        
    def _load_model_thread(self):
        """Load model in separate thread"""
        try:
            self.update_status("Loading model...")
            self.log_message(f"Starting to load {self.current_model_type} model...")
            
            # Load metadata
            data_dir = self.data_dir_var.get()
            
            # Try different metadata files
            metadata_files = ['metadata_mlx_test.pkl', 'metadata.pkl']
            metadata = None
            
            for metadata_file in metadata_files:
                metadata_path = os.path.join(data_dir, metadata_file)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.log_message(f"Loaded metadata from {metadata_file}")
                    break
                    
            if metadata is None:
                raise FileNotFoundError("No metadata file found")
                
            self.metadata = metadata
            
            # Model configuration
            model_kwargs = {
                'n_users': metadata['n_users'],
                'n_items': metadata['n_items'],
                'embedding_dim': int(self.embedding_dim_var.get())
            }
            
            if metadata.get('n_genres', 0) > 0:
                model_kwargs['n_genres'] = metadata['n_genres']
                model_kwargs['n_languages'] = metadata['n_languages']
            
            arch = self.arch_var.get()
            
            if self.current_model_type == "pytorch" and PYTORCH_AVAILABLE:
                self.log_message("Loading PyTorch model...")
                
                # Create model
                model = get_pytorch_model(arch, **model_kwargs)
                
                # Load weights
                model_path = self.pytorch_model_var.get()
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.log_message(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    self.log_message(f"Loaded model weights")
                
                # Create retriever
                self.pytorch_retriever = PyTorchRetriever(model, metadata['n_users'], metadata['n_items'])
                
                # Extract embeddings
                self.log_message("Extracting embeddings...")
                self.pytorch_retriever.extract_embeddings()
                
                # Build index
                self.log_message("Building FAISS index...")
                self.pytorch_retriever.build_index(self.pytorch_retriever.item_embeddings)
                
                self.log_message("PyTorch model loaded successfully!")
                
            elif self.current_model_type == "mlx" and MLX_AVAILABLE:
                self.log_message("Loading MLX model...")
                
                # Create model
                model = get_mlx_model(arch, **model_kwargs)
                
                # Load weights
                model_path = self.mlx_model_var.get()
                model.load_weights(model_path)
                self.log_message(f"Loaded MLX model weights")
                
                # Create retriever
                self.mlx_retriever = MLXEmbeddingRetriever(model, metadata['n_users'], metadata['n_items'])
                
                # Extract embeddings
                self.log_message("Extracting embeddings...")
                self.mlx_retriever.extract_embeddings()
                
                # Build index
                self.log_message("Building FAISS index...")
                self.mlx_retriever.build_index(self.mlx_retriever.item_embeddings)
                
                self.log_message("MLX model loaded successfully!")
                
            else:
                raise ValueError(f"Model type {self.current_model_type} not available")
                
            self.update_status(f"{self.current_model_type} model loaded successfully")
            
        except Exception as e:
            self.log_message(f"Error loading model: {e}")
            self.update_status(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            
        finally:
            # Re-enable load button
            self.load_button.config(state='normal')
            self.is_loading = False
            
    def get_current_retriever(self):
        """Get the current active retriever"""
        if self.current_model_type == "pytorch":
            return self.pytorch_retriever
        elif self.current_model_type == "mlx":
            return self.mlx_retriever
        return None
        
    def get_user_recommendations(self):
        """Get recommendations for a user"""
        retriever = self.get_current_retriever()
        if retriever is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        try:
            user_id = int(self.user_id_var.get())
            top_k = int(self.user_top_k_var.get())
            
            if user_id >= self.metadata['n_users']:
                messagebox.showerror("Error", f"User ID must be less than {self.metadata['n_users']}")
                return
                
            self.update_status(f"Getting recommendations for user {user_id}...")
            
            # Get recommendations
            recommendations = retriever.get_user_recommendations(user_id, k=top_k)
            
            # Clear previous results
            for item in self.user_rec_tree.get_children():
                self.user_rec_tree.delete(item)
                
            # Add new results
            for rank, (item_id, score) in enumerate(recommendations, 1):
                self.user_rec_tree.insert("", "end", values=(rank, item_id, f"{score:.4f}"))
                
            self.update_status(f"Generated {len(recommendations)} recommendations for user {user_id}")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {e}")
            
    def get_item_similarity(self):
        """Get similar items for an item"""
        retriever = self.get_current_retriever()
        if retriever is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        try:
            item_id = int(self.item_id_var.get())
            top_k = int(self.item_top_k_var.get())
            
            if item_id >= self.metadata['n_items']:
                messagebox.showerror("Error", f"Item ID must be less than {self.metadata['n_items']}")
                return
                
            self.update_status(f"Finding similar items for item {item_id}...")
            
            # Get similar items
            similar_items = retriever.get_item_to_item_recommendations(item_id, k=top_k)
            
            # Clear previous results
            for item in self.item_sim_tree.get_children():
                self.item_sim_tree.delete(item)
                
            # Add new results
            for rank, (similar_item_id, score) in enumerate(similar_items, 1):
                self.item_sim_tree.insert("", "end", values=(rank, similar_item_id, f"{score:.4f}"))
                
            self.update_status(f"Found {len(similar_items)} similar items for item {item_id}")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get similar items: {e}")
            
    def batch_process(self):
        """Process batch recommendations"""
        retriever = self.get_current_retriever()
        if retriever is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        # Start batch processing in separate thread
        thread = threading.Thread(target=self._batch_process_thread)
        thread.daemon = True
        thread.start()
        
    def _batch_process_thread(self):
        """Process batch recommendations in separate thread"""
        try:
            n_users = int(self.batch_users_var.get())
            top_k = int(self.batch_top_k_var.get())
            
            self.update_status(f"Processing batch recommendations for {n_users} users...")
            self.progress_label.config(text="Processing...")
            
            retriever = self.get_current_retriever()
            
            # Sample random users
            all_user_ids = list(range(self.metadata['n_users']))
            sample_user_ids = np.random.choice(all_user_ids, min(n_users, len(all_user_ids)), replace=False)
            
            results = {}
            
            for i, user_id in enumerate(sample_user_ids):
                try:
                    recs = retriever.get_user_recommendations(user_id, k=top_k)
                    results[int(user_id)] = [
                        {'item_id': int(item_id), 'score': float(score)}
                        for item_id, score in recs
                    ]
                    
                    # Update progress
                    progress = (i + 1) / len(sample_user_ids) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"Processed {i+1}/{len(sample_user_ids)} users")
                    
                except Exception as e:
                    results[int(user_id)] = []
                    
            # Display results
            self.batch_results_text.delete(1.0, tk.END)
            self.batch_results_text.insert(tk.END, f"Batch Processing Results ({self.current_model_type} model):\n")
            self.batch_results_text.insert(tk.END, f"Processed {len(sample_user_ids)} users\n")
            self.batch_results_text.insert(tk.END, f"Recommendations per user: {top_k}\n\n")
            
            # Show sample results
            for i, (user_id, recs) in enumerate(list(results.items())[:5]):  # Show first 5 users
                self.batch_results_text.insert(tk.END, f"User {user_id}:\n")
                for rank, rec in enumerate(recs[:3], 1):  # Show top 3 recs
                    self.batch_results_text.insert(tk.END, 
                        f"  {rank}. Item {rec['item_id']}: {rec['score']:.4f}\n")
                self.batch_results_text.insert(tk.END, "\n")
                
            if len(results) > 5:
                self.batch_results_text.insert(tk.END, f"... and {len(results) - 5} more users\n")
                
            # Save results
            output_file = f"experiments/gui_batch_results_{self.current_model_type}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.batch_results_text.insert(tk.END, f"\nResults saved to: {output_file}\n")
            
            self.progress_label.config(text="Completed")
            self.update_status(f"Batch processing completed for {len(results)} users")
            
        except Exception as e:
            self.batch_results_text.delete(1.0, tk.END)
            self.batch_results_text.insert(tk.END, f"Error in batch processing: {e}\n")
            self.progress_label.config(text="Error")
            self.update_status(f"Batch processing failed: {e}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = MusicRecommenderGUI(root)
    
    # Check if models are available
    if not PYTORCH_AVAILABLE:
        app.log_message("Warning: PyTorch models not available")
    if not MLX_AVAILABLE:
        app.log_message("Warning: MLX models not available")
        
    if PYTORCH_AVAILABLE or MLX_AVAILABLE:
        app.log_message("GUI initialized successfully")
        app.log_message("Select a model type and click 'Load Model' to begin")
    else:
        app.log_message("Error: No model backends available")
        
    root.mainloop()


if __name__ == "__main__":
    main()