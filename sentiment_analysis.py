"""
Modern Sentiment Analysis Module
This module implements both traditional ML and modern transformer-based approaches
for sentiment analysis of product reviews.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import re
import string
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Modern NLP libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        pipeline, TrainingArguments, Trainer
    )
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Install with: pip install transformers torch")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Install with: pip install nltk")

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set()
        self.lemmatizer = None
        
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except:
                pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text"""
        if not NLTK_AVAILABLE or not self.lemmatizer:
            return text
        
        try:
            tokens = word_tokenize(text)
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized)
        except:
            return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        if not self.stop_words:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess(self, texts: List[str]) -> List[str]:
        """Apply full preprocessing pipeline"""
        processed_texts = []
        
        for text in texts:
            # Clean text
            cleaned = self.clean_text(text)
            
            # Remove stopwords
            cleaned = self.remove_stopwords(cleaned)
            
            # Lemmatize
            cleaned = self.tokenize_and_lemmatize(cleaned)
            
            processed_texts.append(cleaned)
        
        return processed_texts

class TraditionalMLSentimentAnalyzer:
    """Traditional Machine Learning approaches for sentiment analysis"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, text_col: str = 'review_text', 
                    label_col: str = 'sentiment', test_size: float = 0.2):
        """Prepare data for training"""
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess(df[text_col].tolist())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, df[label_col], 
            test_size=test_size, random_state=42, stratify=df[label_col]
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: List[str], y_train: pd.Series):
        """Train multiple traditional ML models"""
        
        # Define models and vectorizers
        model_configs = {
            'tfidf_lr': {
                'vectorizer': TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                'model': LogisticRegression(random_state=42, max_iter=1000)
            },
            'tfidf_svm': {
                'vectorizer': TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                'model': SVC(kernel='linear', random_state=42)
            },
            'tfidf_rf': {
                'vectorizer': TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                'model': RandomForestClassifier(n_estimators=100, random_state=42)
            },
            'tfidf_nb': {
                'vectorizer': TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                'model': MultinomialNB()
            },
            'count_lr': {
                'vectorizer': CountVectorizer(max_features=5000, ngram_range=(1, 2)),
                'model': LogisticRegression(random_state=42, max_iter=1000)
            }
        }
        
        # Train each model
        for name, config in model_configs.items():
            print(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', config['vectorizer']),
                ('classifier', config['model'])
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Store model
            self.models[name] = pipeline
            self.vectorizers[name] = config['vectorizer']
    
    def evaluate_models(self, X_test: List[str], y_test: pd.Series):
        """Evaluate all trained models"""
        self.results = {}
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    def get_best_model(self) -> str:
        """Get the name of the best performing model"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0]
    
    def predict_sentiment(self, text: str, model_name: str = None) -> Dict:
        """Predict sentiment for a single text"""
        if model_name is None:
            model_name = self.get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess([text])[0]
        
        # Predict
        prediction = self.models[model_name].predict([processed_text])[0]
        probability = self.models[model_name].predict_proba([processed_text])[0]
        
        return {
            'prediction': 'positive' if prediction == 1 else 'negative',
            'confidence': float(max(probability)),
            'probabilities': {
                'negative': float(probability[0]),
                'positive': float(probability[1])
            }
        }

class TransformerSentimentAnalyzer:
    """Modern transformer-based sentiment analysis"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            print(f"Loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Fallback to a more common model
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                print("Loaded fallback model: distilbert-base-uncased-finetuned-sst-2-english")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                self.pipeline = None
    
    def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment using transformer model"""
        if self.pipeline is None:
            raise RuntimeError("No model loaded")
        
        # Truncate text if too long
        if len(text) > 512:
            text = text[:512]
        
        # Get predictions
        results = self.pipeline(text)
        
        # Process results
        sentiment_scores = {}
        for result in results[0]:
            label = result['label'].lower()
            score = result['score']
            
            # Map labels to our format
            if 'positive' in label or 'pos' in label:
                sentiment_scores['positive'] = score
            elif 'negative' in label or 'neg' in label:
                sentiment_scores['negative'] = score
            elif 'neutral' in label:
                sentiment_scores['neutral'] = score
        
        # Determine prediction
        if 'positive' in sentiment_scores and 'negative' in sentiment_scores:
            if sentiment_scores['positive'] > sentiment_scores['negative']:
                prediction = 'positive'
                confidence = sentiment_scores['positive']
            else:
                prediction = 'negative'
                confidence = sentiment_scores['negative']
        else:
            # Handle cases where we only have one label
            prediction = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = max(sentiment_scores.values())
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': sentiment_scores
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict_sentiment(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'prediction': 'error',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': str(e)
                })
        return results

class SentimentAnalysisComparison:
    """Compare different sentiment analysis approaches"""
    
    def __init__(self):
        self.traditional_ml = TraditionalMLSentimentAnalyzer()
        self.transformer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformer = TransformerSentimentAnalyzer()
            except Exception as e:
                print(f"Could not initialize transformer model: {e}")
    
    def run_comparison(self, df: pd.DataFrame, text_col: str = 'review_text', 
                      label_col: str = 'sentiment'):
        """Run comprehensive comparison of different approaches"""
        
        print("🚀 Starting Sentiment Analysis Comparison")
        print("=" * 50)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.traditional_ml.prepare_data(
            df, text_col, label_col
        )
        
        # Train traditional ML models
        print("\n📊 Training Traditional ML Models...")
        self.traditional_ml.train_models(X_train, y_train)
        self.traditional_ml.evaluate_models(X_test, y_test)
        
        # Test transformer model
        if self.transformer:
            print("\n🤖 Testing Transformer Model...")
            transformer_results = self.transformer.predict_batch(X_test)
            
            # Convert transformer results to binary predictions
            transformer_predictions = []
            for result in transformer_results:
                if result['prediction'] == 'positive':
                    transformer_predictions.append(1)
                elif result['prediction'] == 'negative':
                    transformer_predictions.append(0)
                else:
                    transformer_predictions.append(1)  # Default to positive for neutral/error
            
            # Calculate transformer metrics
            transformer_accuracy = accuracy_score(y_test, transformer_predictions)
            transformer_f1 = f1_score(y_test, transformer_predictions)
            
            print(f"Transformer: Accuracy={transformer_accuracy:.4f}, F1={transformer_f1:.4f}")
        
        # Summary
        print("\n📈 Results Summary:")
        print("-" * 30)
        
        # Traditional ML results
        for name, results in self.traditional_ml.results.items():
            print(f"{name:15}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
        
        # Transformer results
        if self.transformer:
            print(f"{'Transformer':15}: Accuracy={transformer_accuracy:.4f}, F1={transformer_f1:.4f}")
    
    def create_visualization(self, save_path: str = None):
        """Create comprehensive visualization of results"""
        if not self.traditional_ml.results:
            print("No results to visualize. Run comparison first.")
            return
        
        # Prepare data for visualization
        model_names = list(self.traditional_ml.results.keys())
        accuracies = [self.traditional_ml.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.traditional_ml.results[name]['f1_score'] for name in model_names]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 Score comparison
        bars2 = ax2.bar(model_names, f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # This will be used by the main application
    print("Sentiment Analysis Module Loaded Successfully!")
    print("Available classes:")
    print("- TextPreprocessor: Advanced text preprocessing")
    print("- TraditionalMLSentimentAnalyzer: Traditional ML approaches")
    print("- TransformerSentimentAnalyzer: Modern transformer models")
    print("- SentimentAnalysisComparison: Comprehensive comparison tool")
