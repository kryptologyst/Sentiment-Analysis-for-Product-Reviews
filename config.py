"""
Configuration Management for Sentiment Analysis Project
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "reviews.db"
    backup_path: str = "backups/"
    max_connections: int = 10

@dataclass
class ModelConfig:
    """Model configuration"""
    # Traditional ML models
    traditional_models: list = None
    test_size: float = 0.2
    random_state: int = 42
    
    # Transformer models
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    fallback_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 512
    
    # Text preprocessing
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    
    def __post_init__(self):
        if self.traditional_models is None:
            self.traditional_models = [
                'tfidf_lr',
                'tfidf_svm', 
                'tfidf_rf',
                'tfidf_nb',
                'count_lr'
            ]

@dataclass
class UIConfig:
    """UI configuration"""
    page_title: str = "Sentiment Analysis for Product Reviews"
    page_icon: str = "📊"
    layout: str = "wide"
    theme: str = "light"
    
    # Chart colors
    positive_color: str = "#28a745"
    negative_color: str = "#dc3545"
    neutral_color: str = "#6c757d"
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour

@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig
    model: ModelConfig
    ui: UIConfig
    
    # App settings
    debug: bool = False
    log_level: str = "INFO"
    data_path: str = "data/"
    models_path: str = "models/"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def to_file(self, config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = asdict(self)
        
        if config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration with fallback to defaults"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                return AppConfig.from_file(self.config_path)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        
        # Return default configuration
        return AppConfig(
            database=DatabaseConfig(),
            model=ModelConfig(),
            ui=UIConfig()
        )
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self):
        """Save configuration to file"""
        self.config.to_file(self.config_path)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.data_path,
            self.config.models_path,
            self.config.database.backup_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_env_vars(self) -> Dict[str, Any]:
        """Get environment variables"""
        return {
            'DEBUG': os.getenv('DEBUG', str(self.config.debug)),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', self.config.log_level),
            'DB_PATH': os.getenv('DB_PATH', self.config.database.db_path),
            'MODEL_PATH': os.getenv('MODEL_PATH', self.config.models_path),
            'DATA_PATH': os.getenv('DATA_PATH', self.config.data_path)
        }

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config_manager.get_config()

def update_config(**kwargs):
    """Update the global configuration"""
    config_manager.update_config(**kwargs)

def save_config():
    """Save the global configuration"""
    config_manager.save_config()

# Environment setup
def setup_environment():
    """Setup the application environment"""
    config = get_config()
    
    # Create directories
    config_manager.create_directories()
    
    # Set environment variables
    env_vars = config_manager.get_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Set logging level
    import logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Environment setup complete. Debug mode: {config.debug}")

# Default configuration file content
DEFAULT_CONFIG_YAML = """
database:
  db_path: "reviews.db"
  backup_path: "backups/"
  max_connections: 10

model:
  traditional_models:
    - "tfidf_lr"
    - "tfidf_svm"
    - "tfidf_rf"
    - "tfidf_nb"
    - "count_lr"
  test_size: 0.2
  random_state: 42
  transformer_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  fallback_model: "distilbert-base-uncased-finetuned-sst-2-english"
  max_length: 512
  max_features: 5000
  ngram_range: [1, 2]

ui:
  page_title: "Sentiment Analysis for Product Reviews"
  page_icon: "📊"
  layout: "wide"
  theme: "light"
  positive_color: "#28a745"
  negative_color: "#dc3545"
  neutral_color: "#6c757d"
  cache_ttl: 3600

debug: false
log_level: "INFO"
data_path: "data/"
models_path: "models/"
"""

def create_default_config_file(config_path: str = "config.yaml"):
    """Create a default configuration file"""
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG_YAML)
    print(f"Default configuration file created: {config_path}")

if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    
    # Setup environment
    setup_environment()
    
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Database path: {config.database.db_path}")
    print(f"Model: {config.model.transformer_model}")
    print(f"UI theme: {config.ui.theme}")
