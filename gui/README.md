# Music Recommender GUI

A comprehensive Tkinter-based graphical user interface for demonstrating the ANN (Approximate Nearest Neighbor) search functionality of both PyTorch and MLX music recommendation models.

## Features

### 🎯 **Model Support**
- **PyTorch Models**: Load and use PyTorch-trained recommendation models (.pt files)
- **MLX Models**: Load and use MLX-trained recommendation models (.safetensors files)
- **Automatic Detection**: GUI automatically detects available model types and paths

### 📱 **Four Main Tabs**

#### 1. **Model Setup**
- Select between PyTorch and MLX model types
- Browse and load model files
- Configure model parameters (architecture, embedding dimensions)
- Real-time status updates during model loading

#### 2. **User Recommendations**
- Enter a user ID to get personalized recommendations
- Specify number of recommendations to generate
- View results in an organized table with ranks, item IDs, and scores

#### 3. **Item Similarity**
- Find items similar to a given item
- Useful for "users who liked this also liked" functionality
- Display similarity scores between items

#### 4. **Batch Processing**
- Generate recommendations for multiple users at once
- Progress bar to track batch processing
- Export results to JSON files
- View sample results in the interface

## Installation & Usage

### Prerequisites
```bash
pip install torch mlx-core faiss-cpu sentence-transformers pandas numpy scikit-learn tkinter
```

### Launch the GUI
```bash
# From project root
python run_gui.py

# Or directly
python gui/music_recommender_gui.py
```

## How to Use

### 1. **Load a Model**
1. Go to the "Model Setup" tab
2. Select PyTorch or MLX model type
3. Browse for your model file (or use default paths)
4. Set the data directory path
5. Configure model architecture (NCF, MF, or Hybrid)
6. Click "Load Model" and wait for completion

### 2. **Get User Recommendations**
1. Switch to "User Recommendations" tab
2. Enter a user ID (must be within valid range)
3. Set number of recommendations desired
4. Click "Get Recommendations"
5. View results in the table

### 3. **Find Similar Items**
1. Go to "Item Similarity" tab
2. Enter an item ID
3. Set number of similar items to find
4. Click "Find Similar Items"
5. Browse similarity results

### 4. **Batch Process**
1. Navigate to "Batch Processing" tab
2. Set number of users to process
3. Set recommendations per user
4. Click "Generate Batch Recommendations"
5. Monitor progress and view results

## Model Compatibility

### **PyTorch Models**
- Supports both checkpoint files and final model files
- Automatically detects file format (checkpoint vs. state dict)
- Works with NCF, Matrix Factorization, and Hybrid models

### **MLX Models**
- Uses .safetensors format for optimal performance
- Optimized for Apple Silicon (M1/M2/M3 Macs)
- Same model architectures as PyTorch version

## File Structure
```
gui/
├── music_recommender_gui.py  # Main GUI application
└── README.md                 # This file

run_gui.py                    # Launcher script
```

## Features in Detail

### **Threading**
- Model loading runs in background threads to prevent GUI freezing
- Batch processing uses separate threads with progress updates
- Responsive interface during long operations

### **Error Handling**
- Comprehensive error messages for invalid inputs
- Graceful handling of missing files or incompatible models
- User-friendly warnings and confirmations

### **Auto-Configuration**
- Automatically detects default model paths
- Loads appropriate metadata files for different model types
- Smart fallbacks for missing configuration

### **Export Capabilities**
- Batch results exported to JSON format
- Timestamped filenames to avoid conflicts
- Human-readable result formatting

## Troubleshooting

### **Model Loading Issues**
- Ensure model files exist at specified paths
- Check that metadata files are in the data directory
- Verify model architecture matches the loaded weights

### **Import Errors**
- Install all required dependencies
- Check Python path includes project root
- Ensure both PyTorch and MLX are properly installed

### **Performance Tips**
- Use smaller batch sizes for faster processing
- MLX models typically perform better on Apple Silicon
- PyTorch models work well on both CPU and GPU

## Example Workflow

1. **Start the GUI**: `python run_gui.py`
2. **Load MLX Model**: Select MLX type, use default paths, click Load
3. **Test Single User**: Go to User Recommendations, enter user ID 42, get 10 recommendations
4. **Check Item Similarity**: Switch to Item Similarity, enter item 100, find 5 similar items
5. **Batch Process**: Generate recommendations for 20 users with 5 recommendations each
6. **Export Results**: Results automatically saved to `experiments/gui_batch_results_mlx.json`

The GUI provides an intuitive way to explore and demonstrate the capabilities of your music recommendation system!