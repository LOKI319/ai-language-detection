"""
AI Language Detection Package

This package provides a comprehensive language detection system using
multiple machine learning algorithms including Naive Bayes, Random Forest,
SVM, and Logistic Regression.

Supports detection of 10+ languages including:
English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, and Japanese.
"""

__version__ = "1.0.0"
__author__ = "AI Language Detection Team"

from .language_detector import LanguageDetector

__all__ = ["LanguageDetector"]
