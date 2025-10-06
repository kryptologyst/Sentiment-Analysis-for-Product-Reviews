#!/usr/bin/env python3
"""
Comprehensive Demo Script for Sentiment Analysis Project
This script demonstrates all the features and capabilities of the modernized sentiment analysis system.
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 40)

def demo_basic_analysis():
    """Demonstrate basic sentiment analysis"""
    print_section("Basic Sentiment Analysis")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd
        
        # Sample data
        reviews = [
            "This product is absolutely amazing! Love it!",
            "Terrible quality, waste of money.",
            "Pretty good for the price.",
            "Excellent service and fast delivery.",
            "Not impressed, feels cheap."
        ]
        sentiments = [1, 0, 1, 1, 0]  # 1=positive, 0=negative
        
        # Vectorize
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(reviews)
        
        # Train model
        model = LogisticRegression()
        model.fit(X, sentiments)
        
        # Test on new review
        new_review = "This is fantastic!"
        X_new = vectorizer.transform([new_review])
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0]
        
        print(f"✅ Review: '{new_review}'")
        print(f"✅ Prediction: {'Positive' if prediction == 1 else 'Negative'}")
        print(f"✅ Confidence: {max(probability):.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return False

def demo_mock_database():
    """Demonstrate mock database functionality"""
    print_section("Mock Database Demo")
    
    try:
        from mock_database import MockReviewDatabase
        
        # Create database
        print("📚 Creating mock database...")
        db = MockReviewDatabase("demo_reviews.db")
        
        # Get data
        df = db.get_reviews_dataframe()
        print(f"✅ Database created with {len(df)} reviews")
        print(f"✅ Categories: {df['category'].nunique()}")
        print(f"✅ Products: {df['product_name'].nunique()}")
        
        # Show sample data
        print("\n📋 Sample Reviews:")
        sample_reviews = df[['review_text', 'sentiment', 'product_name', 'category']].head(3)
        for idx, row in sample_reviews.iterrows():
            sentiment = "Positive" if row['sentiment'] == 1 else "Negative"
            print(f"   • {row['review_text'][:50]}... [{sentiment}] - {row['product_name']}")
        
        # Sentiment distribution
        sentiment_dist = db.get_sentiment_distribution()
        print(f"\n📈 Sentiment Distribution:")
        print(f"   • Positive: {sentiment_dist['positive']['count']} ({sentiment_dist['positive']['percentage']}%)")
        print(f"   • Negative: {sentiment_dist['negative']['count']} ({sentiment_dist['negative']['percentage']}%)")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Mock database demo failed: {e}")
        return False

def demo_advanced_analysis():
    """Demonstrate advanced sentiment analysis"""
    print_section("Advanced Sentiment Analysis Demo")
    
    try:
        from sentiment_analysis import SentimentAnalysisComparison, TraditionalMLSentimentAnalyzer
        from mock_database import MockReviewDatabase
        
        # Load data
        print("📚 Loading data...")
        db = MockReviewDatabase("demo_reviews.db")
        df = db.get_reviews_dataframe()
        db.close()
        
        # Run comparison
        print("🤖 Running model comparison...")
        comparison = SentimentAnalysisComparison()
        
        # Use smaller dataset for demo
        demo_df = df.head(100)  # Use first 100 reviews for speed
        comparison.run_comparison(demo_df)
        
        # Show results
        if comparison.traditional_ml.results:
            print("\n📊 Model Performance:")
            for name, results in comparison.traditional_ml.results.items():
                print(f"   • {name}: Accuracy={results['accuracy']:.3f}, F1={results['f1_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced analysis demo failed: {e}")
        return False

def demo_text_preprocessing():
    """Demonstrate text preprocessing capabilities"""
    print_section("Text Preprocessing Demo")
    
    try:
        from sentiment_analysis import TextPreprocessor
        
        # Sample texts
        sample_texts = [
            "This product is AMAZING!!! I love it so much! 😍",
            "Terrible quality... waste of money. Don't buy!",
            "Pretty good for the price, but nothing special.",
            "Check out this link: https://example.com for more info",
            "Contact me at user@email.com for details"
        ]
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        print("📝 Original texts:")
        for i, text in enumerate(sample_texts, 1):
            print(f"   {i}. {text}")
        
        # Process texts
        processed_texts = preprocessor.preprocess(sample_texts)
        
        print("\n✨ Processed texts:")
        for i, text in enumerate(processed_texts, 1):
            print(f"   {i}. {text}")
        
        return True
    except Exception as e:
        print(f"❌ Text preprocessing demo failed: {e}")
        return False

def demo_configuration():
    """Demonstrate configuration management"""
    print_section("Configuration Management Demo")
    
    try:
        from config import get_config, setup_environment
        
        # Setup environment
        setup_environment()
        config = get_config()
        
        print("⚙️  Current Configuration:")
        print(f"   • Database path: {config.database.db_path}")
        print(f"   • Test size: {config.model.test_size}")
        print(f"   • Transformer model: {config.model.transformer_model}")
        print(f"   • UI theme: {config.ui.theme}")
        print(f"   • Debug mode: {config.debug}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration demo failed: {e}")
        return False

def demo_single_prediction():
    """Demonstrate single text prediction"""
    print_section("Single Text Prediction Demo")
    
    try:
        from sentiment_analysis import TraditionalMLSentimentAnalyzer
        from mock_database import MockReviewDatabase
        
        # Load data and train model
        print("📚 Training model...")
        db = MockReviewDatabase("demo_reviews.db")
        df = db.get_reviews_dataframe()
        db.close()
        
        analyzer = TraditionalMLSentimentAnalyzer()
        X_train, X_test, y_train, y_test = analyzer.prepare_data(df.head(50))  # Use small sample
        analyzer.train_models(X_train, y_train)
        analyzer.evaluate_models(X_test, y_test)
        
        # Test predictions
        test_reviews = [
            "This product exceeded my expectations! Highly recommend!",
            "Poor quality, broke after one use. Very disappointed.",
            "It's okay, nothing special but gets the job done.",
            "Amazing quality! Worth every penny.",
            "Terrible customer service and the product doesn't work."
        ]
        
        print("\n🔍 Testing predictions:")
        for review in test_reviews:
            result = analyzer.predict_sentiment(review)
            print(f"   • '{review[:40]}...'")
            print(f"     → {result['prediction'].title()} (confidence: {result['confidence']:.2%})")
        
        return True
    except Exception as e:
        print(f"❌ Single prediction demo failed: {e}")
        return False

def demo_web_interface_info():
    """Show information about web interface"""
    print_section("Web Interface Information")
    
    print("🌐 Streamlit Web Interface Features:")
    print("   • Interactive dashboard with real-time analytics")
    print("   • Model comparison with performance metrics")
    print("   • Single review analysis with confidence scores")
    print("   • Advanced analytics with category filtering")
    print("   • Raw data explorer with export functionality")
    
    print("\n🚀 To launch the web interface:")
    print("   streamlit run app.py")
    print("   Then open: http://localhost:8501")
    
    return True

def cleanup_demo_files():
    """Clean up demo files"""
    print_section("Cleanup")
    
    demo_files = [
        "demo_reviews.db",
        "product_reviews.csv",
        "model_comparison.png"
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Removed: {file}")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")

def main():
    """Main demo function"""
    print_header("Sentiment Analysis Project Demo")
    
    print("🎯 This demo showcases all the features of the modernized sentiment analysis system.")
    print("📋 The demo will run through:")
    print("   • Basic sentiment analysis")
    print("   • Mock database functionality")
    print("   • Advanced text preprocessing")
    print("   • Model comparison")
    print("   • Configuration management")
    print("   • Single text prediction")
    print("   • Web interface information")
    
    input("\n⏸️  Press Enter to start the demo...")
    
    demos = [
        ("Basic Analysis", demo_basic_analysis),
        ("Mock Database", demo_mock_database),
        ("Text Preprocessing", demo_text_preprocessing),
        ("Configuration", demo_configuration),
        ("Single Prediction", demo_single_prediction),
        ("Advanced Analysis", demo_advanced_analysis),
        ("Web Interface Info", demo_web_interface_info)
    ]
    
    successful_demos = 0
    total_demos = len(demos)
    
    for name, demo_func in demos:
        try:
            if demo_func():
                successful_demos += 1
            time.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
    
    # Summary
    print_header("Demo Summary")
    print(f"✅ Successful demos: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
    elif successful_demos > total_demos // 2:
        print("⚠️  Most demos completed successfully with some issues.")
    else:
        print("❌ Several demos failed. Check your installation.")
    
    print("\n📚 Next Steps:")
    print("   • Run 'python 0161.py' for basic analysis")
    print("   • Run 'streamlit run app.py' for web interface")
    print("   • Read README.md for detailed documentation")
    
    # Ask about cleanup
    cleanup_choice = input("\n🧹 Clean up demo files? (y/n): ").lower().strip()
    if cleanup_choice in ['y', 'yes']:
        cleanup_demo_files()

if __name__ == "__main__":
    main()
