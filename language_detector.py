"""
AI Language Detection System
============================

A comprehensive language detection system using multiple machine learning algorithms
including Naive Bayes, Random Forest, SVM, and Logistic Regression.

Supports detection of 10+ languages:
- English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese

Author: AI Language Detection Team
Version: 1.0.0
License: MIT
"""

import os
import re
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Advanced Language Detection System using multiple ML algorithms
    
    This class implements a comprehensive language detection system that can
    identify languages using various machine learning approaches.
    """
    
    def __init__(self, model_type: str = 'naive_bayes'):
        """
        Initialize the Language Detector
        
        Args:
            model_type (str): Type of model to use
                Options: 'naive_bayes', 'random_forest', 'svm', 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # Supported languages mapping
        self.language_codes = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese'
        }
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize model based on type
        self._initialize_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def _initialize_model(self):
        """Initialize the machine learning model based on model_type"""
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=20
            ),
            'svm': SVC(
                kernel='linear', 
                probability=True, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear'
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = models[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for language detection
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep letters and spaces
        text = re.sub(r'[^a-zA-ZÀ-ÿ\u0100-\u017F\u0400-\u04FF\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from texts
        
        Args:
            texts (List[str]): List of texts to extract features from
            
        Returns:
            np.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            # Initialize TF-IDF vectorizer with language-specific parameters
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb',  # Character n-grams
                ngram_range=(1, 3),  # 1-3 character n-grams
                max_features=10000,  # Limit features
                min_df=2,           # Minimum document frequency
                max_df=0.95,        # Maximum document frequency
                sublinear_tf=True   # Use sublinear TF scaling
            )
            
            # Fit and transform
            features = self.vectorizer.fit_transform(texts)
else:
            # Transform only
            features = self.vectorizer.transform(texts)
        
        return features
    
    def train(self, texts: List[str], labels: List[str], 
              test_size: float = 0.2, validation: bool = True):
        """
        Train the language detection model
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Corresponding language labels
            test_size (float): Proportion of data for testing
            validation (bool): Whether to perform cross-validation
        """
        logger.info(f"Training {self.model_type} model with {len(texts)} samples")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract features
        features = self.extract_features(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        if validation:
            cv_scores = cross_val_score(self.model, features, labels, cv=5)
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
        logger."""
AI Language Detection System
============================

A comprehensive language detection system using multiple machine learning algorithms
including Naive Bayes, Random Forest, SVM, and Logistic Regression.

Supports detection of 10+ languages:
- English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese

Author: AI Language Detection Team
Version: 1.0.0
License: MIT
"""

import os
import re
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Advanced Language Detection System using multiple ML algorithms
    
    This class implements a comprehensive language detection system that can
    identify languages using various machine learning approaches.
    """
    
    def __init__(self, model_type: str = 'naive_bayes'):
        """
        Initialize the Language Detector
        
        Args:
            model_type (str): Type of model to use
                Options: 'naive_bayes', 'random_forest', 'svm', 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # Supported languages mapping
        self.language_codes = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese'
        }
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize model based on type
        self._initialize_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def _initialize_model(self):
        """Initialize the machine learning model based on model_type"""
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=20
            ),
            'svm': SVC(
                kernel='linear', 
                probability=True, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear'
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = models[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for language detection
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep letters and spaces
        text = re.sub(r'[^a-zA-ZÀ-ÿ\u0100-\u017F\u0400-\u04FF\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from texts
        
        Args:
            texts (List[str]): List of texts to extract features from
            
        Returns:
            np.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            # Initialize TF-IDF vectorizer with language-specific parameters
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb',  # Character n-grams
                ngram_range=(1, 3),  # 1-3 character n-grams
                max_features=10000,  # Limit features
                min_df=2,           # Minimum document frequency
                max_df=0.95,        # Maximum document frequency
                sublinear_tf=True   # Use sublinear TF scaling
            )
            
            # Fit and transform
            features = self.vectorizer.fit_transform(texts)
        else:
            # Transform only
            features = self.vectorizer.transform(texts)
        
        return features
    
    def train(self, texts: List[str], labels: List[str], 
              test_size: float = 0.2, validation: bool = True):
        """
        Train the language detection model
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Corresponding language labels
            test_size (float): Proportion of data for testing
            validation (bool): Whether to perform cross-validation
        """
        logger.info(f"Training {self.model_type} model with {len(texts)} samples")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract features
        features = self.extract_features(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        if validation:
            cv_scores = cross_val_score(self.model, features, labels, cv=5)
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
        logger.info("Model training completed successfully")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the language of input text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float]: Predicted language code and confidence score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return 'unknown', 0.0
        
        # Extract features
        features = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0  # SVM without probability=True
        
        return prediction, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict languages for multiple texts
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Tuple[str, float]]: List of (language, confidence) tuples
        """
        return [self.predict(text) for text in texts]
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get full language name from language code
        
        Args:
            language_code (str): Language code (e.g., 'en', 'es')
            
        Returns:
            str: Full language name
        """
        return self.language_codes.get(language_code, 'Unknown')
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'language_codes': self.language_codes
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.language_codes = model_data['language_codes']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """
    Create a sample dataset for testing and demonstration
    
    Returns:
        Tuple[List[str], List[str]]: Sample texts and their language labels
    """
    sample_texts = [
        # English
        "Hello, how are you today? I hope you are doing well.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        
        # Spanish
        "Hola, ¿cómo estás hoy? Espero que estés bien.",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "El aprendizaje automático es un campo fascinante de estudio.",
        
        # French
        "Bonjour, comment allez-vous aujourd'hui? J'espère que vous allez bien.",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "L'apprentissage automatique est un domaine d'étude fascinant.",
        
        # German
        "Hallo, wie geht es dir heute? Ich hoffe, es geht dir gut.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Maschinelles Lernen ist ein faszinierendes Studiengebiet.",
    ]
    
    sample_labels = ['en'] * 3 + ['es'] * 3 + ['fr'] * 3 + ['de'] * 3
    
    return sample_texts, sample_labels


def main():
    """
    Main function for demonstration and testing
    """
    print("AI Language Detection System")
    print("=" * 40)
    
    # Create sample dataset
    texts, labels = create_sample_dataset()
    
    # Test different model types
    model_types = ['naive_bayes', 'random_forest', 'logistic_regression']
    
    for model_type in model_types:
        print(f"\nTesting {model_type.replace('_', ' ').title()} Model:")
        print("-" * 50)
        
        try:
            # Initialize detector
            detector = LanguageDetector(model_type=model_type)
            
            # Train model
            detector.train(texts, labels, validation=False)
            
            # Test predictions
            test_texts = [
                "Good morning, everyone!",
                "Buenos días a todos!",
                "Bonjour tout le monde!",
                "Guten Morgen, alle zusammen!"
            ]
            
            print("\nPredictions:")
            for text in test_texts:
                lang_code, confidence = detector.predict(text)
                lang_name = detector.get_language_name(lang_code)
                print(f"Text: '{text}'")
                print(f"Predicted: {lang_name} ({lang_code}) - Confidence: {confidence:.3f}")
                print()
                
        except Exception as e:
            logger.error(f"Error with {model_type}: {str(e)}")


if __name__ == "__main__":
    main()("Model training completed successfully")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the language of input text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float]: Predicted language code and confidence score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return 'unknown', 0.0
        
        # Extract features
        features = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
else:
            confidence = 1.0  # SVM without probability=True
        
        return prediction, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict languages for multiple texts
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Tuple[str, float]]: List of (language, confidence) tuples
        """
        return [self.predict(text) for text in texts]
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get full language name from language code
        
        Args:
            language_code (str): Language code (e.g., 'en', 'es')
            
        Returns:
            str: Full language name
        """
        return self.language_codes.get(language_code, 'Unknown')
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'language_codes': self.language_codes
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.language_codes = model_data['language_codes']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """
    Create a sample dataset for testing and demonstration
    
    Returns:
        Tuple[List[str], List[str]]: Sample texts and their language labels
    """
    sample_texts = [
        # English
        "Hello, how are you today? I hope you are doing well.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        
        # Spanish
        "Hola, ¿cómo estás hoy? Espero que estés bien.",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "El aprendizaje automático es un campo fascinante de estudio.",
        
        # French
        "Bonjour, comment allez-vous aujourd'hui? J'espère que vous allez bien.",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "L'apprentissage automatique est un domaine d'étude fascinant.",
        
        # German
        "Hallo, wie geht es dir heute? Ich hoffe, es geht dir gut.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Maschinelles Lernen ist ein faszinierendes
