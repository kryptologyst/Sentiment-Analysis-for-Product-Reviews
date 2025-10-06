"""
Modern Sentiment Analysis Web Application
A comprehensive Streamlit-based web UI for product review sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json

# Import our custom modules
from mock_database import MockReviewDatabase
from sentiment_analysis import (
    SentimentAnalysisComparison, 
    TraditionalMLSentimentAnalyzer,
    TransformerSentimentAnalyzer,
    TextPreprocessor
)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis for Product Reviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_database():
    """Load the mock database"""
    try:
        db = MockReviewDatabase()
        df = db.get_reviews_dataframe()
        db.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

@st.cache_data
def get_sentiment_distribution(df):
    """Get sentiment distribution from dataframe"""
    if df is None:
        return None
    
    sentiment_counts = df['sentiment'].value_counts()
    total = len(df)
    
    return {
        'positive': {
            'count': sentiment_counts.get(1, 0),
            'percentage': round((sentiment_counts.get(1, 0) / total) * 100, 2)
        },
        'negative': {
            'count': sentiment_counts.get(0, 0),
            'percentage': round((sentiment_counts.get(0, 0) / total) * 100, 2)
        }
    }

def create_sentiment_distribution_chart(df):
    """Create sentiment distribution pie chart"""
    sentiment_dist = get_sentiment_distribution(df)
    
    if sentiment_dist is None:
        return None
    
    labels = ['Positive', 'Negative']
    values = [sentiment_dist['positive']['count'], sentiment_dist['negative']['count']]
    colors = ['#28a745', '#dc3545']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_traces(
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=14
    )
    fig.update_layout(
        title="Sentiment Distribution",
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    return fig

def create_category_analysis_chart(df):
    """Create category-wise sentiment analysis chart"""
    if df is None or df.empty:
        return None
    
    # Group by category and sentiment
    category_sentiment = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    
    # Calculate percentages
    category_percentages = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    
    # Add positive sentiment bars
    fig.add_trace(go.Bar(
        name='Positive',
        x=category_percentages.index,
        y=category_percentages[1],
        marker_color='#28a745',
        text=[f'{val:.1f}%' for val in category_percentages[1]],
        textposition='auto'
    ))
    
    # Add negative sentiment bars
    fig.add_trace(go.Bar(
        name='Negative',
        x=category_percentages.index,
        y=category_percentages[0],
        marker_color='#dc3545',
        text=[f'{val:.1f}%' for val in category_percentages[0]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Sentiment Distribution by Product Category",
        xaxis_title="Product Category",
        yaxis_title="Percentage (%)",
        barmode='stack',
        height=500,
        font=dict(size=12)
    )
    
    return fig

def create_rating_sentiment_chart(df):
    """Create rating vs sentiment analysis chart"""
    if df is None or df.empty:
        return None
    
    # Create correlation between rating and sentiment
    rating_sentiment = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    # Add bars for each rating
    for rating in sorted(df['rating'].unique()):
        if rating in rating_sentiment.index:
            positive_count = rating_sentiment.loc[rating, 1] if 1 in rating_sentiment.columns else 0
            negative_count = rating_sentiment.loc[rating, 0] if 0 in rating_sentiment.columns else 0
            
            fig.add_trace(go.Bar(
                name=f'{rating} Stars',
                x=['Positive', 'Negative'],
                y=[positive_count, negative_count],
                marker_color='lightblue' if rating >= 4 else 'lightcoral',
                text=[positive_count, negative_count],
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Rating Distribution by Sentiment",
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        barmode='group',
        height=400,
        font=dict(size=12)
    )
    
    return fig

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis for Product Reviews</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading product review data..."):
        df = load_database()
    
    if df is None:
        st.error("Failed to load data. Please check the database connection.")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Data overview
    st.sidebar.markdown("### 📈 Data Overview")
    st.sidebar.metric("Total Reviews", len(df))
    st.sidebar.metric("Product Categories", df['category'].nunique())
    st.sidebar.metric("Unique Products", df['product_name'].nunique())
    
    # Sentiment distribution in sidebar
    sentiment_dist = get_sentiment_distribution(df)
    if sentiment_dist:
        st.sidebar.markdown("### 🎯 Sentiment Distribution")
        st.sidebar.metric(
            "Positive Reviews", 
            f"{sentiment_dist['positive']['count']} ({sentiment_dist['positive']['percentage']}%)"
        )
        st.sidebar.metric(
            "Negative Reviews", 
            f"{sentiment_dist['negative']['count']} ({sentiment_dist['negative']['percentage']}%)"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🤖 Model Comparison", 
        "🔍 Single Review Analysis", 
        "📈 Advanced Analytics", 
        "📋 Raw Data"
    ])
    
    with tab1:
        st.header("📊 Sentiment Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        
        with col2:
            positive_count = sentiment_dist['positive']['count'] if sentiment_dist else 0
            st.metric("Positive Reviews", positive_count)
        
        with col3:
            negative_count = sentiment_dist['negative']['count'] if sentiment_dist else 0
            st.metric("Negative Reviews", negative_count)
        
        with col4:
            avg_rating = df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_chart = create_sentiment_distribution_chart(df)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
        
        with col2:
            category_chart = create_category_analysis_chart(df)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)
        
        # Rating analysis
        rating_chart = create_rating_sentiment_chart(df)
        if rating_chart:
            st.plotly_chart(rating_chart, use_container_width=True)
    
    with tab2:
        st.header("🤖 Model Comparison")
        
        st.markdown("Compare different sentiment analysis models and approaches.")
        
        if st.button("🚀 Run Model Comparison", type="primary"):
            with st.spinner("Training and comparing models..."):
                # Initialize comparison
                comparison = SentimentAnalysisComparison()
                
                # Run comparison
                comparison.run_comparison(df)
                
                # Display results
                st.success("Model comparison completed!")
                
                # Show results in a nice format
                if comparison.traditional_ml.results:
                    st.subheader("📊 Model Performance Results")
                    
                    # Create results dataframe
                    results_data = []
                    for name, results in comparison.traditional_ml.results.items():
                        results_data.append({
                            'Model': name.replace('_', ' ').title(),
                            'Accuracy': f"{results['accuracy']:.4f}",
                            'F1 Score': f"{results['f1_score']:.4f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Create visualization
                    model_names = list(comparison.traditional_ml.results.keys())
                    accuracies = [comparison.traditional_ml.results[name]['accuracy'] for name in model_names]
                    f1_scores = [comparison.traditional_ml.results[name]['f1_score'] for name in model_names]
                    
                    # Accuracy chart
                    fig_acc = px.bar(
                        x=model_names, 
                        y=accuracies,
                        title="Model Accuracy Comparison",
                        labels={'x': 'Model', 'y': 'Accuracy'},
                        color=accuracies,
                        color_continuous_scale='Blues'
                    )
                    fig_acc.update_layout(height=400)
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # F1 Score chart
                    fig_f1 = px.bar(
                        x=model_names, 
                        y=f1_scores,
                        title="Model F1 Score Comparison",
                        labels={'x': 'Model', 'y': 'F1 Score'},
                        color=f1_scores,
                        color_continuous_scale='Reds'
                    )
                    fig_f1.update_layout(height=400)
                    st.plotly_chart(fig_f1, use_container_width=True)
    
    with tab3:
        st.header("🔍 Single Review Analysis")
        
        st.markdown("Analyze the sentiment of individual product reviews.")
        
        # Text input
        review_text = st.text_area(
            "Enter a product review to analyze:",
            height=100,
            placeholder="Enter your product review here..."
        )
        
        # Model selection
        model_type = st.selectbox(
            "Select Analysis Method:",
            ["Traditional ML (Best Model)", "Transformer Model", "Both"]
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary") and review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                results = {}
                
                # Traditional ML analysis
                if model_type in ["Traditional ML (Best Model)", "Both"]:
                    try:
                        traditional_analyzer = TraditionalMLSentimentAnalyzer()
                        X_train, X_test, y_train, y_test = traditional_analyzer.prepare_data(df)
                        traditional_analyzer.train_models(X_train, y_train)
                        traditional_analyzer.evaluate_models(X_test, y_test)
                        
                        best_model = traditional_analyzer.get_best_model()
                        traditional_result = traditional_analyzer.predict_sentiment(review_text, best_model)
                        results['Traditional ML'] = traditional_result
                    except Exception as e:
                        st.error(f"Traditional ML analysis failed: {e}")
                
                # Transformer analysis
                if model_type in ["Transformer Model", "Both"]:
                    try:
                        transformer_analyzer = TransformerSentimentAnalyzer()
                        transformer_result = transformer_analyzer.predict_sentiment(review_text)
                        results['Transformer'] = transformer_result
                    except Exception as e:
                        st.error(f"Transformer analysis failed: {e}")
                
                # Display results
                if results:
                    st.success("Analysis completed!")
                    
                    for model_name, result in results.items():
                        st.subheader(f"📊 {model_name} Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sentiment_class = result['prediction']
                            sentiment_color = "positive-sentiment" if sentiment_class == "positive" else "negative-sentiment"
                            st.markdown(f"**Sentiment:** <span class='{sentiment_color}'>{sentiment_class.title()}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            confidence = result['confidence']
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        with col3:
                            if 'probabilities' in result:
                                prob_text = f"Positive: {result['probabilities'].get('positive', 0):.2%}<br>Negative: {result['probabilities'].get('negative', 0):.2%}"
                                st.markdown(f"**Probabilities:**<br>{prob_text}", unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Advanced Analytics")
        
        st.markdown("Deep dive into sentiment patterns and trends.")
        
        # Category filter
        categories = ['All'] + list(df['category'].unique())
        selected_category = st.selectbox("Filter by Category:", categories)
        
        # Filter data
        filtered_df = df if selected_category == 'All' else df[df['category'] == selected_category]
        
        # Time-based analysis
        st.subheader("📅 Time-based Sentiment Trends")
        
        # Convert review_date to datetime
        filtered_df['review_date'] = pd.to_datetime(filtered_df['review_date'])
        
        # Monthly sentiment trends
        monthly_sentiment = filtered_df.groupby([
            filtered_df['review_date'].dt.to_period('M'), 'sentiment'
        ]).size().unstack(fill_value=0)
        
        if not monthly_sentiment.empty:
            fig_monthly = px.line(
                monthly_sentiment.reset_index(),
                x='review_date',
                y=[0, 1],
                title=f"Monthly Sentiment Trends - {selected_category}",
                labels={'value': 'Number of Reviews', 'review_date': 'Month'}
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Brand analysis
        st.subheader("🏷️ Brand Sentiment Analysis")
        
        brand_sentiment = filtered_df.groupby(['brand', 'sentiment']).size().unstack(fill_value=0)
        brand_percentages = brand_sentiment.div(brand_sentiment.sum(axis=1), axis=0) * 100
        
        if not brand_percentages.empty:
            fig_brand = px.bar(
                brand_percentages.reset_index(),
                x='brand',
                y=[0, 1],
                title=f"Brand Sentiment Distribution - {selected_category}",
                labels={'value': 'Percentage (%)', 'brand': 'Brand'}
            )
            fig_brand.update_layout(height=400)
            st.plotly_chart(fig_brand, use_container_width=True)
        
        # Price vs Sentiment analysis
        st.subheader("💰 Price vs Sentiment Analysis")
        
        # Create price bins
        filtered_df['price_range'] = pd.cut(
            filtered_df['price'], 
            bins=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        price_sentiment = filtered_df.groupby(['price_range', 'sentiment']).size().unstack(fill_value=0)
        price_percentages = price_sentiment.div(price_sentiment.sum(axis=1), axis=0) * 100
        
        if not price_percentages.empty:
            fig_price = px.bar(
                price_percentages.reset_index(),
                x='price_range',
                y=[0, 1],
                title=f"Price Range vs Sentiment - {selected_category}",
                labels={'value': 'Percentage (%)', 'price_range': 'Price Range'}
            )
            fig_price.update_layout(height=400)
            st.plotly_chart(fig_price, use_container_width=True)
    
    with tab5:
        st.header("📋 Raw Data Explorer")
        
        st.markdown("Explore the raw product review data.")
        
        # Data filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category_filter = st.selectbox("Category:", ['All'] + list(df['category'].unique()))
        
        with col2:
            sentiment_filter = st.selectbox("Sentiment:", ['All', 'Positive', 'Negative'])
        
        with col3:
            rating_filter = st.selectbox("Rating:", ['All'] + [str(i) for i in sorted(df['rating'].unique())])
        
        # Apply filters
        filtered_data = df.copy()
        
        if category_filter != 'All':
            filtered_data = filtered_data[filtered_data['category'] == category_filter]
        
        if sentiment_filter != 'All':
            sentiment_value = 1 if sentiment_filter == 'Positive' else 0
            filtered_data = filtered_data[filtered_data['sentiment'] == sentiment_value]
        
        if rating_filter != 'All':
            filtered_data = filtered_data[filtered_data['rating'] == int(rating_filter)]
        
        # Display filtered data
        st.subheader(f"📊 Filtered Data ({len(filtered_data)} reviews)")
        
        # Data table
        display_columns = ['review_text', 'sentiment', 'rating', 'product_name', 'category', 'brand', 'price']
        st.dataframe(
            filtered_data[display_columns], 
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
