import re
import urllib.parse
import numpy as np
import tldextract

class URLFeatureExtractor:
    def __init__(self):
        pass
    
    # This function extract all features from the URL
    def extract():
        pass
    
    # This function extract features from multiple URLs
    def extract_batch():
        pass

    # This function calculates shannon entropy
    def calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        
        prob_dict = {}
        for char in text:
            prob_dict[char] = prob_dict.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        for count in prob_dict.values():
            prob = count / text_len
            entropy -= prob * np.log2(prob)

        return entropy

    # This function is for counting the number of subdomains
    def count_subdomains(self, url: str) -> int:
        try:
            ext = tldextract.extract(url)
            if ext.subdomain:
                return len(ext.subdomain.split('.'))
            return 0
        except:
            try:
                parsed = urllib.parse.urlparse(url)
                domain_parts = parsed.netloc.split('.')
                return max(0, len(domain_parts) - 2)
            except:
                return 0

    # This function gets the length of domain length
    def get_domain_length(self, url: str) -> int:
        try:
            parsed = urllib.parse.urlparse(url)
            return len(parsed.netloc)
        except:
            return 0

    # This function gets the length of path
    def get_path_length(self, url: str) -> int:
        try:
            parsed = urllib.parse.urlparse(url)
            return len(parsed.path)
        except:
            return 0

    # This function checks if the URL is using the shortening service
    def has_shortening_service(self, url: str) -> int:
        shortners = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 't.co', 'is.gd', 'buff.ly']
        return 1 if any(short in url.lower() for short in shortners) else 0

    # This function checks if the domain name appears in path
    def domain_in_path(self, url: str) -> int:
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            # Check if any part of domain appears in path
            domain_parts = domain.split('.')
            for part in domain_parts:
                if len(part) > 3 and part in path:
                    return 1
            return 0
        except:
            return 0

    def count_special_character():
        pass
    
    # This function checks for suspicous port number
    def has_suspicious_port():
        pass

    # This function checks for prefix / suffix seperators in domain
    def prefix_suffix_in_domain():
        pass
    
    # This function detects the abnormal URL pattern
    def abnormal_url():
        pass

    # This function gets the name of all extracted features
    def get_features_name():
        pass
