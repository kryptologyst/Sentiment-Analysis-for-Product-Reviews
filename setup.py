#!/usr/bin/env python3
"""
Setup script for Sentiment Analysis Project
This script helps users set up the project environment and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "models", 
        "backups",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_nltk_data():
    """Download required NLTK data"""
    print("📚 Setting up NLTK data...")
    
    try:
        import nltk
        nltk_data = [
            'stopwords',
            'punkt', 
            'wordnet',
            'vader_lexicon'
        ]
        
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"✅ Downloaded NLTK data: {data}")
            except Exception as e:
                print(f"⚠️  Warning: Could not download {data}: {e}")
        
        return True
    except ImportError:
        print("⚠️  NLTK not available, skipping NLTK data download")
        return False

def create_config_file():
    """Create default configuration file"""
    print("⚙️  Creating configuration file...")
    
    try:
        from config import create_default_config_file
        create_default_config_file()
        print("✅ Configuration file created")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not create config file: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib.pyplot as plt
        print("✅ Basic packages imported successfully")
        
        # Test our modules
        try:
            from mock_database import MockReviewDatabase
            from sentiment_analysis import SentimentAnalysisComparison
            from config import get_config
            print("✅ Custom modules imported successfully")
        except ImportError as e:
            print(f"⚠️  Warning: Some custom modules not available: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\n📋 Usage Instructions:")
    print("\n1. Basic Analysis:")
    print("   python 0161.py")
    
    print("\n2. Web Interface:")
    print("   streamlit run app.py")
    print("   Then open: http://localhost:8501")
    
    print("\n3. Generate Mock Data:")
    print("   python mock_database.py")
    
    print("\n4. Advanced Analysis:")
    print("   python sentiment_analysis.py")
    
    print("\n5. Configuration:")
    print("   python config.py")
    
    print("\n📚 Documentation:")
    print("   See README.md for detailed information")
    
    print("\n🔧 Configuration:")
    print("   Edit config.yaml to customize settings")

def main():
    """Main setup function"""
    print("🚀 Sentiment Analysis Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Create config file
    create_config_file()
    
    # Test installation
    if test_installation():
        print_usage_instructions()
    else:
        print("❌ Setup completed with warnings")
        print("Some features may not work properly")

if __name__ == "__main__":
    main()
