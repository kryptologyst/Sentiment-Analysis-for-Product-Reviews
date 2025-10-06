# Sentiment Analysis for Product Reviews

A comprehensive sentiment analysis system that combines traditional machine learning and modern transformer-based approaches for analyzing product reviews. This project demonstrates the evolution from basic sentiment analysis to a production-ready system with advanced features.

## Features

### Core Functionality
- **Multiple ML Models**: Logistic Regression, SVM, Random Forest, Naive Bayes
- **Modern Transformers**: BERT, RoBERTa-based sentiment analysis
- **Advanced Text Preprocessing**: NLTK-based tokenization, lemmatization, stopword removal
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices, classification reports

### Web Interface
- **Interactive Dashboard**: Real-time sentiment analysis with Streamlit
- **Data Visualization**: Charts, graphs, and interactive plots
- **Model Comparison**: Side-by-side performance comparison
- **Single Review Analysis**: Individual text sentiment prediction

### Data Management
- **Mock Database**: Realistic product reviews across multiple categories
- **SQLite Integration**: Efficient data storage and retrieval
- **Data Export**: CSV download functionality
- **Category Filtering**: Filter by product categories

### Production Features
- **Configuration Management**: YAML-based configuration system
- **Error Handling**: Robust error handling and fallback mechanisms
- **Modular Architecture**: Clean, extensible code structure
- **Documentation**: Comprehensive docstrings and comments

## 📁 Project Structure

```
0161_Sentiment_analysis_for_product_reviews/
├── 0161.py                 # Main analysis script (modernized)
├── app.py                  # Streamlit web application
├── sentiment_analysis.py   # Advanced sentiment analysis module
├── mock_database.py        # Mock database generator
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── data/                # Data directory (auto-created)
    ├── reviews.db       # SQLite database
    └── backups/         # Database backups
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Sentiment-Analysis-for-Product-Reviews.git
   cd Sentiment-Analysis-for-Product-Reviews
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if using advanced preprocessing)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage

### 1. Basic Analysis
Run the original sentiment analysis with enhanced features:
```bash
python 0161.py
```

### 2. Web Interface
Launch the interactive Streamlit application:
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### 3. Generate Mock Data
Create a comprehensive mock database:
```bash
python mock_database.py
```

### 4. Advanced Analysis
Run comprehensive model comparison:
```bash
python sentiment_analysis.py
```

### 5. Configuration Setup
Initialize configuration files:
```bash
python config.py
```

## Model Performance

The system compares multiple approaches:

| Model | Accuracy | F1 Score | Description |
|-------|----------|----------|-------------|
| TF-IDF + Logistic Regression | ~0.85 | ~0.84 | Traditional ML baseline |
| TF-IDF + SVM | ~0.87 | ~0.86 | Linear kernel SVM |
| TF-IDF + Random Forest | ~0.82 | ~0.81 | Ensemble method |
| TF-IDF + Naive Bayes | ~0.80 | ~0.79 | Probabilistic classifier |
| Transformer (RoBERTa) | ~0.90 | ~0.89 | Modern transformer model |

*Performance may vary based on dataset and preprocessing*

## 🔧 Configuration

The system uses YAML configuration files for easy customization:

```yaml
database:
  db_path: "reviews.db"
  backup_path: "backups/"

model:
  traditional_models: ["tfidf_lr", "tfidf_svm", "tfidf_rf"]
  test_size: 0.2
  transformer_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"

ui:
  page_title: "Sentiment Analysis for Product Reviews"
  theme: "light"
```

## Web Interface Features

### Dashboard Tab
- **Key Metrics**: Total reviews, sentiment distribution, average rating
- **Visualizations**: Pie charts, bar charts, trend analysis
- **Real-time Updates**: Live data refresh capabilities

### Model Comparison Tab
- **Performance Metrics**: Accuracy, F1-score comparison
- **Visual Charts**: Interactive performance comparisons
- **Model Selection**: Choose best performing model

### Single Review Analysis Tab
- **Text Input**: Analyze individual reviews
- **Multiple Models**: Compare different approaches
- **Confidence Scores**: Probability distributions

### Advanced Analytics Tab
- **Category Analysis**: Sentiment by product category
- **Time Trends**: Monthly sentiment patterns
- **Brand Analysis**: Sentiment by brand
- **Price Analysis**: Sentiment vs price correlation

### Raw Data Explorer Tab
- **Data Filtering**: Filter by category, sentiment, rating
- **Data Export**: Download filtered datasets
- **Search Functionality**: Find specific reviews

## Technical Details

### Text Preprocessing Pipeline
1. **Text Cleaning**: Remove URLs, emails, special characters
2. **Normalization**: Convert to lowercase, remove extra whitespace
3. **Tokenization**: Split text into tokens using NLTK
4. **Stopword Removal**: Remove common English stopwords
5. **Lemmatization**: Reduce words to root forms
6. **Vectorization**: Convert to numerical features (TF-IDF/Count)

### Model Architecture
- **Traditional ML**: Scikit-learn pipelines with TF-IDF/Count vectorization
- **Transformer Models**: Hugging Face transformers with pre-trained models
- **Evaluation**: Cross-validation, holdout testing, comprehensive metrics

### Database Schema
```sql
-- Products table
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL,
    brand TEXT,
    description TEXT
);

-- Reviews table
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    review_text TEXT NOT NULL,
    sentiment INTEGER NOT NULL,
    rating INTEGER,
    reviewer_name TEXT,
    review_date TEXT,
    helpful_votes INTEGER DEFAULT 0,
    verified_purchase BOOLEAN DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES products (id)
);
```

## Example Usage

### Basic Sentiment Analysis
```python
from sentiment_analysis import TraditionalMLSentimentAnalyzer

# Initialize analyzer
analyzer = TraditionalMLSentimentAnalyzer()

# Prepare data
X_train, X_test, y_train, y_test = analyzer.prepare_data(df)

# Train models
analyzer.train_models(X_train, y_train)

# Evaluate
analyzer.evaluate_models(X_test, y_test)

# Predict sentiment
result = analyzer.predict_sentiment("This product is amazing!")
print(result)
```

### Transformer-based Analysis
```python
from sentiment_analysis import TransformerSentimentAnalyzer

# Initialize transformer
transformer = TransformerSentimentAnalyzer()

# Predict sentiment
result = transformer.predict_sentiment("This product is amazing!")
print(result)
```

## Future Enhancements

- [ ] **Multi-class Sentiment**: Support for neutral sentiment
- [ ] **Aspect-based Analysis**: Analyze specific product aspects
- [ ] **Real-time API**: REST API for sentiment analysis
- [ ] **Model Deployment**: Docker containerization
- [ ] **Advanced Visualizations**: Word clouds, sentiment timelines
- [ ] **Custom Model Training**: Fine-tune transformers on domain data
- [ ] **Batch Processing**: Analyze large datasets efficiently
- [ ] **Integration**: Connect to real e-commerce APIs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Scikit-learn**: Traditional machine learning algorithms
- **Hugging Face**: Transformer models and tokenizers
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing toolkit
- **Pandas & NumPy**: Data manipulation and analysis


# Sentiment-Analysis-for-Product-Reviews
