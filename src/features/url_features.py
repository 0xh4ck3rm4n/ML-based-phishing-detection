import re
import urllib.parse
from typing import List
import numpy as np
import tldextract

class URLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'login', 'signin', 'account', 'update', 'verify', 'secure',
            'banking', 'confirm', 'password', 'suspend', 'urgent', 'click',
            'free', 'bonus', 'prize', 'winner', 'congratulations'
        ]
        
        self.legitimate_tlds = ['.com', '.org', '.edu', '.gov', '.net']
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click']
    
    # This function extract all features from the URL
    def extract(self, url: str) -> np.ndarray:
        features = []
        
        # Basic length features
        features.append(len(url))                           
        features.append(url.count('.'))                     
        features.append(url.count('-'))                     
        features.append(url.count('_'))                     
        features.append(url.count('/'))                     
        features.append(url.count('?'))                    
        features.append(url.count('='))                    
        features.append(url.count('@'))                    
        features.append(url.count('&'))                     
        features.append(sum(c.isdigit() for c in url))
        
        # Binary features
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        features.append(1 if re.search(ip_pattern, url) else 0)  # 11. Has IP
        features.append(1 if url.startswith('https://') else 0)   # 12. Uses HTTPS
        
        # Keyword analysis
        keyword_count = sum(1 for kw in self.suspicious_keywords if kw in url.lower())
        features.append(keyword_count)                      # 13. Suspicious keyword count
        
        # TLD analysis
        has_legit_tld = any(tld in url.lower() for tld in self.legitimate_tlds)
        features.append(0 if has_legit_tld else 1)         
        
        has_suspicious_tld = any(tld in url.lower() for tld in self.suspicious_tlds)
        features.append(1 if has_suspicious_tld else 0)    
        
        # Entropy
        features.append(self._calculate_entropy(url))       
        
        # Domain features
        features.append(self._count_subdomains(url))        
        features.append(self._get_domain_length(url))       
        features.append(self._get_path_length(url))         
        
        # Advanced features
        features.append(self._has_shortening_service(url)) 
        features.append(self._domain_in_path(url))          
        features.append(self._count_special_chars(url))     
        features.append(self._has_suspicious_port(url))     
        features.append(self._prefix_suffix_in_domain(url)) 
        features.append(self._abnormal_url(url))            
        
        return np.array(features, dtype=float)
    
    # This function extract features from multiple URLs
    def extract_batch(self, urls: List[str]) -> np.ndarray:
        return np.array([self.extract(url) for url in urls])

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
            domain_parts = domain.split('.')
            for part in domain_parts:
                if len(part) > 3 and part in path:
                    return 1
            return 0
        except:
            return 0

    # This function is to count special characters
    def count_special_character(self, url: str) -> int:
        special_chars = ['!', '#', '$', '%', '^', '*', '(', ')', '[', ']', '{', '}', '|', '\\', '<', '>']
        return sum(url.count(char) for char in special_chars)
    
    # This function checks for suspicous port number
    def has_suspicious_port(self, url: str) -> int:
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.port and parsed.port not in [80, 443, 8080, 8443]:
                return 1
            return 0
        except:
            return 0

    # This function checks for prefix / suffix seperators in domain
    def prefix_suffix_in_domain(self, url: str) -> int:
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            return 1 if '-' in domain or '_' in domain else 0
        except:
            return 0
    
    # This function detects the abnormal URL pattern
    def abnormal_url(self, url: str) -> int:
        try:
            if url.count('http') > 1:
                return 1
            parsed = urllib.parse.urlparse(url)
            if '@' in parsed.netloc:
                return 1
            return 0
        except:
            return 0

    # This function gets the name of all extracted features
    def get_features_name(self) -> List[str]:
        return [
            'url_length', 'dot_count', 'hyphen_count', 'underscore_count',
            'slash_count', 'question_count', 'equal_count', 'at_count',
            'ampersand_count', 'digit_count', 'has_ip', 'uses_https',
            'suspicious_keyword_count', 'has_suspicious_tld', 'has_bad_tld',
            'entropy', 'subdomain_count', 'domain_length', 'path_length',
            'has_url_shortener', 'domain_in_path', 'special_char_count',
            'suspicious_port', 'prefix_suffix_domain', 'abnormal_url'
        ]

if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    test_urls = ["https://www.google.com/search", "http://paypal-verify.tk/login.php?id=123&session=abc"]

    for url in test_urls:
        features = extractor.extract(url)
        print(f"\nURL: {url}")
        print(f"Features: {features}")
        print(f"Feature count: {len(features)}")