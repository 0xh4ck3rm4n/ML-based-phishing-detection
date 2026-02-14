import os 
import pandas as pd

class DatasetLoader:

    def __init__(self, data_dir: str = './datasets'):
        """
        Args:
            data_dir : directory containing dataset files 
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    # This function load kaggle phishing site urls dataset
    def load_kaggle_phishing_urls(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'phishing_site_urls.csv')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("Kaggale dataset not found at {filepath}")
        
        print(f"Loading Kaggle Phishing URLs from {filepath}")
        df = pd.read_csv(filepath)

        url_col = 'URL' if 'URL' in df.columns else df.columns[0]
        label_col = 'Label' if "Label" in df.columns else df.columns[1]

        urls = df[url_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()

        labels = self.normalize_labels(labels_raw)

        print(f"Loaded {len(urls)} URLs")  
        print(f" - Legitimate: {labels.count(0)}")
        print(f" - Phishing: {labels.count(1)}")

        return urls, labels

    # This function normalize various label format to 0 [legitimate] and 1 [phishing]
    def normalize_labels():
        pass      
