from typing import List, Dict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class TextFeatureExtractor:
    
    def __init__(self, max_tfidf_features: int = 500, use_transformers: bool = False):
        self.phishing_keywords = [
            'urgent', 'verify', 'suspended', 'click here', 'confirm',
            'account', 'security', 'update', 'password', 'immediate',
            'winner', 'congratulations', 'prize', 'claim', 'act now',
            'limited time', 'expire', 'suspend', 'verify identity',
            'unusual activity', 'unauthorized', 'blocked', 'refund',
            'tax', 'irs', 'social security', 'ssn', 'credit card',
            'bank account', 'wire transfer', 'western union', 'paypal'
        ]
        
        self.urgency_words = [
            'urgent', 'immediate', 'now', 'asap', 'today', 'quickly',
            'hurry', 'fast', 'limited', 'expires', 'deadline', 'act now'
        ]
        
        self.money_words = [
            'money', 'cash', 'dollar', 'prize', 'winner', 'reward',
            'bonus', 'free', 'earn', 'income', 'profit', 'credit'
        ]
        
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.use_transformers = use_transformers
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        if use_transformers:
            self._init_transformer()
        
        self.is_fitted = False
    
    # This function initializes transformer model (BERT) for embeddings
    def _init_transformer(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = 'distilbert-base-uncased'
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            print(f"✓ Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"⚠️  Could not load transformer model: {e}")
            self.use_transformers = False
    
    # This function fit the TF-IDF vectorizer on training texts
    def fit(self, texts: List[str]):
        self.tfidf.fit(texts)
        self.is_fitted = True
    
    # This function extract statistical features from text
    def extract_statistical():
        pass

    # This function extract TF-IDF features from texts
    def extract_tfidf(self, texts: List[str]):
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit() first.")

        return self.tfidf.transform(texts)
    
    # This function is used to extract embeddings using transformer model (BERT)
    def extract_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.use_transformers or self.transformer_model is None:
            raise ValueError("Transformer model not initialized")
        
        import torch
        
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.transformer_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    # This function extract combined features: statistical + TF-IDF
    def extract_combined():
        pass
    
    # This function get names of statistical features
    def get_feature_names(self) -> List[str]:
        return [
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'exclamation_ratio',
            'capital_count', 'capital_ratio', 'all_caps_words',
            'phishing_keyword_count', 'urgency_word_count', 'money_word_count',
            'has_click_here', 'has_verify', 'has_urgent',
            'url_count', 'email_count',
            'sentiment_positive', 'sentiment_negative', 'sentiment_compound'
        ]