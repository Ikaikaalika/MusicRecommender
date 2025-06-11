#!/usr/bin/env python3
"""
Music Recommender GUI Launcher
Simple script to launch the music recommendation GUI
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the GUI
try:
    from gui.music_recommender_gui import main
    print("Launching Music Recommender GUI...")
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install torch mlx faiss-cpu sentence-transformers pandas numpy scikit-learn")
except Exception as e:
    print(f"Error launching GUI: {e}")