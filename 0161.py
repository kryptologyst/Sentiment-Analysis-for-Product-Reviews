"""
Project 161: Modern Sentiment Analysis for Product Reviews
========================================================

A comprehensive sentiment analysis system that combines traditional machine learning
and modern transformer-based approaches for analyzing product reviews.

This modernized version includes:
- Advanced text preprocessing with NLTK
- Multiple ML models (Logistic Regression, SVM, Random Forest, Naive Bayes)
- Modern transformer models (BERT, RoBERTa)
- Comprehensive evaluation and comparison
- Interactive web interface with Streamlit
- Mock database with realistic product reviews
- Configuration management system

Usage:
    python 0161.py                    # Run basic analysis
    python app.py                      # Launch web interface
    python mock_database.py            # Generate mock data
    python sentiment_analysis.py       # Run advanced analysis
    python config.py                   # Setup configuration

Requirements:
    pip install -r requirements.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from mock_database import MockReviewDatabase
    from sentiment_analysis import SentimentAnalysisComparison
    from config import setup_environment, get_config
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Some modules not available: {e}")
    print("Running in basic mode...")
    MODULES_AVAILABLE = False

def run_basic_analysis():
    """Run the original basic sentiment analysis"""
    print("🚀 Running Basic Sentiment Analysis")
    print("=" * 50)
    
    # Sample dataset (enhanced version)
    data = {
        'review': [
            "Absolutely love this product! Works perfectly and exceeded my expectations.",
            "Worst purchase I've ever made. Do not recommend to anyone.",
            "Pretty decent quality for the price. Good value for money.",
            "It broke after one use. Terrible quality and poor construction.",
            "Great value for money. Very satisfied with this purchase!",
            "The item never arrived. Bad experience with shipping and customer service.",
            "Highly recommend! Exceeded my expectations in every way.",
            "It was okay, nothing special but gets the job done.",
            "Cheap material, not worth the price at all. Very disappointed.",
            "Fantastic product! Will definitely buy again and recommend to friends.",
            "Excellent quality! This product is worth every penny.",
            "Poor customer service and the product doesn't work as advertised.",
            "Amazing! Best purchase I've made this year.",
            "Not impressed. The product feels cheap and unreliable.",
            "Outstanding quality! This exceeded all my expectations."
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Dataset loaded: {len(df)} reviews")
    print(f"📈 Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], 
        test_size=0.3, random_state=42, stratify=df['sentiment']
    )
    
    print(f"📚 Training set: {len(X_train)} reviews")
    print(f"🧪 Test set: {len(X_test)} reviews")
    
    # Text preprocessing with TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Multiple models for comparison
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    results = {}
    
    print("\n🤖 Training Models...")
    print("-" * 30)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"✅ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_model = models[best_model_name]
    best_predictions = results[best_model_name]['predictions']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"📊 F1 Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Detailed classification report
    print(f"\n📋 Classification Report ({best_model_name}):")
    print("-" * 40)
    print(classification_report(y_test, best_predictions, target_names=["Negative", "Positive"]))
    
    # Confusion matrix visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Model comparison chart
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return results

def run_advanced_analysis():
    """Run advanced analysis using our custom modules"""
    if not MODULES_AVAILABLE:
        print("❌ Advanced modules not available. Install requirements first.")
        return None
    
    print("🚀 Running Advanced Sentiment Analysis")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    config = get_config()
    
    # Load mock database
    print("📚 Loading mock database...")
    db = MockReviewDatabase(config.database.db_path)
    df = db.get_reviews_dataframe()
    db.close()
    
    print(f"📊 Loaded {len(df)} reviews from {df['category'].nunique()} categories")
    
    # Run comprehensive comparison
    comparison = SentimentAnalysisComparison()
    comparison.run_comparison(df)
    
    # Create visualizations
    comparison.create_visualization("model_comparison.png")
    
    return comparison

def main():
    """Main function to run the sentiment analysis"""
    print("🎯 Sentiment Analysis for Product Reviews")
    print("=" * 60)
    print("Choose analysis mode:")
    print("1. Basic Analysis (Original)")
    print("2. Advanced Analysis (Modern)")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_basic_analysis()
        elif choice == "2":
            run_advanced_analysis()
        elif choice == "3":
            print("\n" + "="*60)
            run_basic_analysis()
            print("\n" + "="*60)
            run_advanced_analysis()
        else:
            print("Invalid choice. Running basic analysis...")
            run_basic_analysis()
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Running basic analysis as fallback...")
        run_basic_analysis()

if __name__ == "__main__":
    main()

# 🧠 What This Modernized Project Demonstrates:
# ✅ Advanced text preprocessing with NLTK (tokenization, lemmatization, stopword removal)
# ✅ Multiple ML models comparison (Logistic Regression, SVM, Random Forest, Naive Bayes)
# ✅ Modern transformer-based sentiment analysis (BERT, RoBERTa)
# ✅ Comprehensive evaluation metrics and visualizations
# ✅ Interactive web interface with Streamlit
# ✅ Mock database with realistic product reviews across multiple categories
# ✅ Configuration management system
# ✅ Production-ready code structure with error handling
# ✅ Extensible architecture for easy model addition