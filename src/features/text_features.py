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
            print(f"Loaded transformer model: {model_name}")
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            self.use_transformers = False
    
    # This function fit the TF-IDF vectorizer on training texts
    def fit(self, texts: List[str]):
        self.tfidf.fit(texts)
        self.is_fitted = True
    
    # This function extract statistical features from text
    def extract_statistical(self, text: str) -> Dict[str, float]:
        features = {}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        features['text_length'] = len(text)                                 
        features['word_count'] = len(words)                                 
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0 
        
        features['exclamation_count'] = text.count('!')                     
        features['question_count'] = text.count('?')                        
        features['exclamation_ratio'] = text.count('!') / max(len(text), 1) 
        
        features['capital_count'] = sum(1 for c in text if c.isupper())   
        features['capital_ratio'] = features['capital_count'] / max(len(text), 1)  
        features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)  
        
        features['phishing_keyword_count'] = sum(1 for kw in self.phishing_keywords if kw in text_lower)  
        features['urgency_word_count'] = sum(1 for uw in self.urgency_words if uw in text_lower)  
        features['money_word_count'] = sum(1 for mw in self.money_words if mw in text_lower)  
        
        features['has_click_here'] = 1 if 'click here' in text_lower else 0  
        features['has_verify'] = 1 if 'verify' in text_lower else 0          
        features['has_urgent'] = 1 if 'urgent' in text_lower else 0        
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        features['url_count'] = len(re.findall(url_pattern, text))          
        features['email_count'] = len(re.findall(email_pattern, text))      
        
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_positive'] = sentiment_scores['pos']            
        features['sentiment_negative'] = sentiment_scores['neg']            
        features['sentiment_compound'] = sentiment_scores['compound']       
        
        return features

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
            inputs = self.transformer_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    # This function extract combined features: statistical + TF-IDF
    def extract_combined(self, texts: List[str]) -> np.ndarray:
        stat_features = []
        for text in texts:
            stat_dict = self.extract_statistical(text)
            stat_features.append(list(stat_dict.values()))
        
        stat_features = np.array(stat_features)
        
        tfidf_features = self.extract_tfidf(texts).toarray()
        
        combined = np.hstack([stat_features, tfidf_features])
        
        return combined
    
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
    
class EmailFeatureExtractor(TextFeatureExtractor):
    
    # This function is used for extracting specific email features
    def extract_email_specific(self, email: str) -> Dict[str, float]:
        features = {}
        
        features['has_subject'] = 1 if 'subject:' in email.lower() else 0
        features['has_from'] = 1 if 'from:' in email.lower() else 0
        features['has_to'] = 1 if 'to:' in email.lower() else 0
        
        features['has_attachment_mention'] = 1 if any(word in email.lower() for word in ['attachment', 'attached', 'file']) else 0
        features['has_link_text'] = 1 if 'click' in email.lower() and 'here' in email.lower() else 0
        
        return features

class SMSFeatureExtractor(TextFeatureExtractor):

    # This function is used for extracting SMS specific features
    def extract_sms_specific(self, sms: str) -> Dict[str, float]:
        features = {}
        
        features['has_short_code'] = 1 if re.search(r'\b\d{5,6}\b', sms) else 0
        features['has_premium_rate'] = 1 if re.search(r'\b(send|text|reply)\s+\w+\s+to\s+\d+', sms, re.I) else 0
        features['message_length'] = len(sms)
        features['is_very_short'] = 1 if len(sms) < 20 else 0
        
        features['has_opt_out'] = 1 if any(word in sms.lower() for word in ['stop', 'unsubscribe', 'opt out']) else 0
        
        return features

if __name__ == "__main__":
    extractor = TextFeatureExtractor()
    
    test_texts = [
        "Hello, your order has been shipped.",
        "URGENT!!! Click here NOW to verify your account!!!"
    ]
    
    extractor.fit(test_texts)
    
    for text in test_texts:
        features = extractor.extract_statistical(text)
        print(f"\nText: {text[:50]}...")
        print(f"Features: {features}")