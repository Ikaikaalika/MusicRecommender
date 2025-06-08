# Music Recommendation System with KKBOX Dataset

## Overview
This project implements a music recommendation engine using collaborative filtering, matrix factorization, and embedding-based similarity search. The system is designed to demonstrate skills relevant to Smule's personalized content recommendation needs.

## Features
- Collaborative filtering using matrix factorization
- Embedding-based similarity search with FAISS
- Hybrid retrieval-reranking pipeline
- Evaluation using Mean Reciprocal Rank (MRR)
- Optimized for Apple Silicon (M1) with PyTorch

## Requirements
```bash
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.4
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
```

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install requirements
pip install -r requirements.txt

# For M1 Mac, ensure you have the ARM64 version of PyTorch
pip install torch torchvision torchaudio
```

## Dataset
Download the KKBOX dataset from Kaggle:
1. Go to https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
2. Download and extract to the `data/` directory
3. Ensure you have:
   - `train.csv`: User-song interaction data
   - `songs.csv`: Song metadata
   - `test.csv`: Test data (optional)
   - `members.csv`: User metadata (optional)

## Project Structure
```
music_recommender/
├── data/
│   ├── train.csv
│   ├── songs.csv
│   ├── test.csv
│   └── members.csv
├── scripts/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── ann_search.py
├── notebooks/
│   └── eda.ipynb
├── output/
│   ├── recommendations.csv
│   ├── embeddings/
│   └── models/
├── requirements.txt
└── README.md
```

## Usage

### 1. Data Preprocessing
```bash
python scripts/preprocess.py --input_dir data/ --output_dir output/
```

### 2. Training
```bash
python scripts/train.py --epochs 50 --batch_size 1024 --embedding_dim 128
```

### 3. Generate Recommendations
```bash
python scripts/ann_search.py --user_id 12345 --top_k 10
```

### 4. Evaluation
```bash
python scripts/evaluate.py --test_file data/test.csv
```

## Model Architecture
- **Matrix Factorization**: Neural collaborative filtering with user and item embeddings
- **Embedding Dimension**: 128 (configurable)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary Cross-Entropy for implicit feedback

## Performance Metrics
- Mean Reciprocal Rank (MRR)
- Hit Rate @K
- NDCG (Normalized Discounted Cumulative Gain)

## Optimization for M1 Mac
- Uses PyTorch with Metal Performance Shaders (MPS) backend when available
- Batch processing optimized for 16GB RAM
- Efficient data loading with memory mapping