"""
Mock Database for Product Reviews Sentiment Analysis
This module creates a comprehensive dataset of realistic product reviews
across different categories with proper sentiment labels.
"""

import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

class MockReviewDatabase:
    def __init__(self, db_path: str = "reviews.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        self.populate_database()
    
    def create_tables(self):
        """Create database tables for reviews and products"""
        cursor = self.conn.cursor()
        
        # Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL,
                brand TEXT,
                description TEXT
            )
        ''')
        
        # Reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                review_text TEXT NOT NULL,
                sentiment INTEGER NOT NULL,  -- 0: negative, 1: positive
                rating INTEGER,  -- 1-5 stars
                reviewer_name TEXT,
                review_date TEXT,
                helpful_votes INTEGER DEFAULT 0,
                verified_purchase BOOLEAN DEFAULT 0,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        self.conn.commit()
    
    def populate_database(self):
        """Populate database with realistic product reviews"""
        # Check if data already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] > 0:
            return  # Database already populated
        
        # Product categories and sample products
        products_data = [
            # Electronics
            {"name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 89.99, "brand": "TechSound", "description": "Premium wireless headphones with noise cancellation"},
            {"name": "Smartphone Case", "category": "Electronics", "price": 24.99, "brand": "ProtectPro", "description": "Durable silicone case with raised edges"},
            {"name": "USB-C Charging Cable", "category": "Electronics", "price": 12.99, "brand": "PowerFlow", "description": "Fast charging cable with braided design"},
            {"name": "Wireless Mouse", "category": "Electronics", "price": 35.99, "brand": "ClickTech", "description": "Ergonomic wireless mouse with RGB lighting"},
            
            # Home & Kitchen
            {"name": "Coffee Maker", "category": "Home & Kitchen", "price": 129.99, "brand": "BrewMaster", "description": "Programmable coffee maker with thermal carafe"},
            {"name": "Air Fryer", "category": "Home & Kitchen", "price": 79.99, "brand": "CrispyCook", "description": "Digital air fryer with multiple cooking functions"},
            {"name": "Blender", "category": "Home & Kitchen", "price": 59.99, "brand": "SmoothBlend", "description": "High-speed blender for smoothies and soups"},
            {"name": "Non-stick Pan Set", "category": "Home & Kitchen", "price": 45.99, "brand": "CookEasy", "description": "3-piece non-stick cookware set"},
            
            # Clothing
            {"name": "Cotton T-Shirt", "category": "Clothing", "price": 19.99, "brand": "ComfortWear", "description": "Soft cotton t-shirt in various colors"},
            {"name": "Running Shoes", "category": "Clothing", "price": 89.99, "brand": "SpeedRun", "description": "Lightweight running shoes with cushioned sole"},
            {"name": "Winter Jacket", "category": "Clothing", "price": 149.99, "brand": "WarmGuard", "description": "Insulated winter jacket with hood"},
            {"name": "Jeans", "category": "Clothing", "price": 39.99, "brand": "DenimStyle", "description": "Classic fit jeans in multiple washes"},
            
            # Books
            {"name": "Programming Book", "category": "Books", "price": 34.99, "brand": "TechPress", "description": "Complete guide to Python programming"},
            {"name": "Cookbook", "category": "Books", "price": 24.99, "brand": "CulinaryArts", "description": "Healthy recipes for everyday cooking"},
            {"name": "Fiction Novel", "category": "Books", "price": 14.99, "brand": "StoryTeller", "description": "Bestselling mystery thriller"},
            {"name": "Self-Help Book", "category": "Books", "price": 18.99, "brand": "LifeGuide", "description": "Guide to personal development and success"},
        ]
        
        # Insert products
        cursor.executemany('''
            INSERT INTO products (name, category, price, brand, description)
            VALUES (?, ?, ?, ?, ?)
        ''', [(p["name"], p["category"], p["price"], p["brand"], p["description"]) for p in products_data])
        
        # Generate realistic reviews for each product
        review_templates = {
            "positive": [
                "Absolutely love this product! Exceeded my expectations and works perfectly.",
                "Great quality and fast shipping. Highly recommend to anyone looking for this type of product.",
                "Perfect! Exactly what I was looking for. Great value for money.",
                "Excellent product! Works exactly as described and arrived quickly.",
                "Outstanding quality! Will definitely buy again and recommend to friends.",
                "Fantastic product! Easy to use and great results. Very satisfied!",
                "Amazing quality! This product is worth every penny. Love it!",
                "Perfect purchase! Great product, fast delivery, excellent customer service.",
                "Wonderful product! Exceeded my expectations in every way.",
                "Top quality! This is exactly what I needed. Highly satisfied customer.",
                "Excellent! Great product, great price, great service. Five stars!",
                "Love it! This product is fantastic and I use it every day.",
                "Outstanding! Great quality and exactly as advertised. Highly recommend!",
                "Perfect! This product is amazing and I'm so glad I bought it.",
                "Excellent quality! This product is worth the investment. Love it!"
            ],
            "negative": [
                "Terrible product! Broke after just one use. Complete waste of money.",
                "Poor quality and doesn't work as advertised. Very disappointed.",
                "Worst purchase I've ever made. Do not recommend to anyone.",
                "Cheaply made and stopped working after a few days. Avoid this product.",
                "Awful quality! This product is a complete rip-off. Save your money.",
                "Disappointed with this purchase. Product doesn't meet expectations at all.",
                "Poor construction and unreliable. Would not buy again.",
                "Terrible experience! Product arrived damaged and customer service was unhelpful.",
                "Waste of money! This product is not worth the price at all.",
                "Very poor quality! This product failed to meet basic expectations.",
                "Disappointing purchase. Product quality is much lower than expected.",
                "Not recommended! This product is overpriced and underperforms.",
                "Bad product! Doesn't work properly and feels cheaply made.",
                "Regret this purchase. Product is not as described and poor quality.",
                "Avoid this product! Poor quality and terrible customer service."
            ]
        }
        
        # Generate reviews for each product
        for product_id in range(1, len(products_data) + 1):
            # Generate 15-25 reviews per product
            num_reviews = random.randint(15, 25)
            
            for _ in range(num_reviews):
                # 70% positive, 30% negative (realistic distribution)
                sentiment = 1 if random.random() < 0.7 else 0
                sentiment_key = "positive" if sentiment == 1 else "negative"
                
                review_text = random.choice(review_templates[sentiment_key])
                rating = random.randint(4, 5) if sentiment == 1 else random.randint(1, 2)
                
                # Generate random reviewer name
                first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sage", "River"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
                reviewer_name = f"{random.choice(first_names)} {random.choice(last_names)}"
                
                # Generate random date within last year
                start_date = datetime.now() - timedelta(days=365)
                random_date = start_date + timedelta(days=random.randint(0, 365))
                
                helpful_votes = random.randint(0, 15)
                verified_purchase = random.choice([0, 1])
                
                cursor.execute('''
                    INSERT INTO reviews (product_id, review_text, sentiment, rating, reviewer_name, review_date, helpful_votes, verified_purchase)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (product_id, review_text, sentiment, rating, reviewer_name, random_date.strftime("%Y-%m-%d"), helpful_votes, verified_purchase))
        
        self.conn.commit()
    
    def get_reviews_dataframe(self) -> pd.DataFrame:
        """Get all reviews as a pandas DataFrame"""
        query = '''
            SELECT r.review_text, r.sentiment, r.rating, r.reviewer_name, r.review_date,
                   p.name as product_name, p.category, p.brand, p.price
            FROM reviews r
            JOIN products p ON r.product_id = p.id
        '''
        return pd.read_sql_query(query, self.conn)
    
    def get_reviews_by_category(self, category: str) -> pd.DataFrame:
        """Get reviews filtered by product category"""
        query = '''
            SELECT r.review_text, r.sentiment, r.rating, r.reviewer_name, r.review_date,
                   p.name as product_name, p.category, p.brand, p.price
            FROM reviews r
            JOIN products p ON r.product_id = p.id
            WHERE p.category = ?
        '''
        return pd.read_sql_query(query, self.conn, params=(category,))
    
    def get_sentiment_distribution(self) -> Dict:
        """Get sentiment distribution statistics"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sentiment, COUNT(*) FROM reviews GROUP BY sentiment")
        results = cursor.fetchall()
        
        total_reviews = sum(count for _, count in results)
        distribution = {}
        for sentiment, count in results:
            sentiment_label = "positive" if sentiment == 1 else "negative"
            distribution[sentiment_label] = {
                "count": count,
                "percentage": round((count / total_reviews) * 100, 2)
            }
        
        return distribution
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Sample usage and data export
if __name__ == "__main__":
    # Create database
    db = MockReviewDatabase()
    
    # Get all reviews
    df = db.get_reviews_dataframe()
    print(f"Total reviews: {len(df)}")
    print(f"Sentiment distribution: {db.get_sentiment_distribution()}")
    
    # Save to CSV for easy access
    df.to_csv("product_reviews.csv", index=False)
    print("Reviews saved to product_reviews.csv")
    
    # Show sample data
    print("\nSample reviews:")
    print(df[['review_text', 'sentiment', 'product_name', 'category']].head(10))
    
    db.close()
