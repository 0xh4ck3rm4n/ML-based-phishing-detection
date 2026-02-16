from typing import List, Dict

class TextFeatureExtractor:
    
    def __init__():
        pass
    
    # This function initializes transformer model (BERT) for embeddings
    def _init_transformer():
        pass
    
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
    def extract_transformer_embeddings():
        pass
    
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