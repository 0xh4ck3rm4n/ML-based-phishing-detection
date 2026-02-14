import numpy as np

class ULRFeatureExtractor:

    def __init__(self):
        self.sus_keywords = ['urgent', 'required', 'invoice', 'account update', 'password', 'confirm', 'lottery', 'login', 'signin', 'banking', 'money', 'action', 'file', 'request']

        self.good_tlds = ['.com', '.gov', '.org', '.edu', '.net']

    #[TODO: This function extracts numerical and symbolic features from URL]
    def extract_feature():
        pass

    # This function calculates the shannon entropy of the text
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
