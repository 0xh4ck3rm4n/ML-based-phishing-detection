import os 
import pandas as pd
import numpy as np
import json
import warnings
import glob
from typing import Tuple, List, Optional, Dict
warnings.filterwarnings('ignore')

class DatasetLoader:

    def __init__(self, data_dir: str = './datasets'):
        """
        Args:
            data_dir : directory containing dataset files 
        """
        self.data_dir = data_dir
        self.email_dir = os.path.join(data_dir, 'emails')
        self.sms_dir = os.path.join(data_dir, 'sms')
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.video_dir = os.path.join(data_dir, 'video')
        self.deepfake_dir = os.path.join(data_dir, 'deepfake')
        
        for directory in [self.email_dir, self.sms_dir, self.audio_dir, 
                         self.video_dir, self.deepfake_dir]:
            os.makedirs(directory, exist_ok=True)
    
    # This function load kaggle phishing site urls dataset
    def load_kaggle_phishing_urls(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if filepath is None:
            filepath = os.path.join(self.email_dir, 'phishing_site_urls.csv')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Kaggale dataset not found at {filepath}")
        
        print(f"Loading Kaggle Phishing URLs from {filepath}")
        df = pd.read_csv(filepath)

        url_col = 'URL' if 'URL' in df.columns else df.columns[0]
        label_col = 'Label' if "Label" in df.columns else df.columns[1]

        urls = df[url_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()

        labels = self.normalize_labels(labels_raw)

        print(f"Loaded {len(urls)} URLs")  
        print(f"    - Legitimate: {labels.count(0)}")
        print(f"    - Phishing: {labels.count(1)}")

        return urls, labels
    

    # This function loads kaggle phishing email dataset
    def load_kaggle_phishing_emails(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if filepath is None:
            filepath = os.path.join(self.email_dir, 'Phishing_Email.csv')

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Kaggle dataset not found at {filepath}")
        
        print(f"Loading Kaggle Phishing Email From {filepath}")
        df = pd.read_csv(filepath)

        text_col = next((col for col in df.columns if 'email' in col.lower() or 'text' in col.lower() or 'message' in col.lower()), df.columns[0])
        label_col = next((col for col in df.columns if 'label' in col.lower() or 'type' in col.lower()), df.columns[1])

        emails = df[text_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()

        labels = self.normalize_labels(labels_raw)

        print(f"Loading {len(emails)} emails")
        print(f"    - Legitimate: {labels.count(0)}")
        print(f"    - Phishing: {labels.count(1)}")

    # This function loads kaggle spam sms dataset
    def load_sms_spam_collection(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if filepath is None:
            filepath = os.path.join(self.sms_dir, 'spam.csv')
        
        if not os.path.exists(filepath):
            print(f"SMS spam dataset not found at {filepath}")
            return [], []
        
        print(f"Loading SMS Spam Collection...")
        
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath)
        
        if 'v1' in df.columns and 'v2' in df.columns:
            label_col, text_col = 'v1', 'v2'
        else:
            label_col = df.columns[0]
            text_col = df.columns[1]
        
        messages = df[text_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()
        
        labels = [1 if 'spam' in str(label).lower() else 0 for label in labels_raw]
        
        print(f"Loaded {len(messages)} SMS messages")
        print(f"  - Legitimate: {labels.count(0)}")
        print(f"  - Spam/Phishing: {labels.count(1)}")
        
        return messages, labels

    # This function is for loading audio deepfake dataset
    def load_audio_deepfake_dataset(self, audio_dir: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if audio_dir is None:
            audio_dir = self.audio_dir
        
        real_dir = os.path.join(audio_dir, 'real')
        fake_dir = os.path.join(audio_dir, 'fake')

        audio_paths = []
        labels = []

        # Load real audio files
        if os.path.exists(real_dir):
            real_files = glob.glob(os.path.join(real_dir, '*.wav')) + glob.glob(os.path.join(real_dir, '*mp3')) + glob.glob(os.path.join(real_dir, '*flac'))

            audio_paths.extend(real_files)
            labels.extend([0] * len(real_files))
            print(f"Found {len(real_files)} real audio files!")
        
        # Load fake audio files
        if os.path.exists(fake_dir):
            fake_files = glob.glob(os.path.join(fake_dir, '*.wav')) + glob.glob(os.path.join(fake_dir, '*.mp3')) + glob.glob(os.path.join(fake_dir, '*.flac'))

            audio_paths.extend(fake_files)
            labels.extend([1] * len(fake_files))
            print(f"Found {len(fake_files)} fake audio files!")

        if not audio_paths:
            print(f"No audio files found in {audio_dir}")
            print(f"Expected structure: {audio_dir}/real/ and {audio_dir}/fake/")

        return audio_paths, labels

    # This function is for loading deepfake video dataset
    def load_faceforensics_dataset(self, video_dir: Optional[str] = None) -> Tuple[List[str], List[int]]:
        if video_dir is None:
            video_dir = self.video_dir
        
        original_dir = os.path.join(video_dir, 'original')
        manipulate_dir = os.path.join(video_dir, 'manipulated')

        video_paths = []
        labels = []

        # Load real video files
        if os.path.exists(original_dir):
            original_files = glob.glob(os.path.join(original_dir, '*.mp4')) + glob.glob(os.path.join(original_dir, '*.avi'))

            video_paths.extend(original_files)
            labels.extend([0] * len(original_files))
            print(f"Found {len(original_files)} original video files!")
        
        # Load fake video files
        if os.path.exists(manipulate_dir):
            manipulate_files = glob.glob(os.path.join(manipulate_dir, '*.mp4')) + glob.glob(os.path.join(manipulate_dir, '*.avi'))

            video_paths.extend(manipulate_files)
            labels.extend([1] * len(manipulate_files))
            print(f"Found {len(manipulate_files)} manipulated video files!")

        if not video_paths:
            print(f"No video files found in {video_paths}")
            print(f"Expected structure: {video_paths}/original/ and {video_paths}/manipulate/")
        
        return video_paths, labels


    ###############################
    #        HELPER FUNCTION      #
    ###############################

    # This function normalize various label format to 0 [legitimate] and 1 [phishing]
    def normalize_labels(self, labels: List) -> List[int]:
        """
        Args:
            labels: list of labels in various format
        
        Returns:
            list of 0s and 1s
        """
        normalized = []

        for label in labels:
            label_str = str(label).lower().strip()
            if any(term in label_str for term in ['phishing', 'bad', 'malicious', 'spam', '1', 'true']):
                normalized.append(1)
            elif any(term in label_str for term in ['legitimate', 'good', 'safe', '0', 'false']):
                normalized.append(0)
            else:
                try:
                    normalized.append(1 if int(float(label)) > 0 else 0)
                except:
                    normalized.append(0)
        
        return normalized
    
    # This function generates legitimate URL to balance dataset
    def generate_legitimate_url(self, count: int) -> List[str]:
        legitimate_domains = ['google.com', 'wikipedia.org', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'reddit.com', 'amazon.com', 'ebay.com',
            'github.com', 'stackoverflow.com', 'apple.com', 'microsoft.com', 'netflix.com',
            'cnn.com', 'bbc.com', 'nytimes.com', 'medium.com', 'wordpress.org',
            'gmail.com', 'yahoo.com', 'bing.com', 'adobe.com', 'paypal.com']
        
        paths = ['', '/about', '/contact', '/products', '/services', '/blog', '/news', '/help', '/support']

        urls = []
        for i in range(count):
            domain = legitimate_domains[i % len(legitimate_domains)]
            path = paths[i % len(paths)]
            urls.append(f"https://www.{domain}{path}")

        return urls
    
    # This function loads all the available text based dataset
    def load_all_text_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        print("LOADING ALL TEXT-BASED DATASETS")
        print("="*60 + "\n")

        data = {}
        
        try:
            urls, url_labels = self.load_kaggle_phishing_urls()
            data['urls'] = (urls, url_labels)
        except FileNotFoundError as e:
            print(f"Skipping URLs {str(e)}")
            data['urls'] = ([], [])

        try:
            emails, email_labels = self.load_kaggle_phishing_emails()
            data['emails'] = (emails, email_labels)
        except FileNotFoundError as e:
            print(f"Skipping emails {str(e)}")
            data['emails'] = ([], [])

        try:
            sms, sms_labels = self.load_sms_spam_collection()
            data['sms'] = (sms, sms_labels)
        except FileNotFoundError as e:
            print(f"Skipping SMS {str(e)}")
            data['sms'] = ([], [])
        
        return data
    
    # This function loads all the available audio and video dataset
    def load_all_audio_video_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        print("LOADING ALL AUDIO/VIDEO DATASETS")
        print("="*60 + "\n")
        
        data = {}
        
        # Load Audio
        audio_paths, audio_labels = self.load_audio_deepfake_dataset()
        data['audio'] = (audio_paths, audio_labels)
        
        # Load Video
        video_paths, video_labels = self.load_faceforensics_dataset()
        data['video'] = (video_paths, video_labels)
        
        return data
    
    # This function provides the summary of all the available dataset
    def get_dataset_summary(self) -> Dict:
        text_data = self.load_all_text_data()
        av_data = self.load_all_audio_video_data()
        
        summary = {
            'urls': len(text_data['urls'][0]),
            'emails': len(text_data['emails'][0]),
            'sms': len(text_data['sms'][0]),
            'audio': len(av_data['audio'][0]),
            'video': len(av_data['video'][0]),
            'total': sum([
                len(text_data['urls'][0]),
                len(text_data['emails'][0]),
                len(text_data['sms'][0]),
                len(av_data['audio'][0]),
                len(av_data['video'][0])
            ])
        }
        return summary

if __name__ == "__main__":
    loader = DatasetLoader()

    summary = loader.get_dataset_summary()
    print("DATASET SUMMARY")
    print("="*60)
    print(f"URLs:   {summary['urls']:,}")
    print(f"Emails: {summary['emails']:,}")
    print(f"SMS:    {summary['sms']:,}")
    print(f"Audio:  {summary['audio']:,}")
    print(f"Video:  {summary['video']:,}")
    print(f"─" * 60)
    print(f"TOTAL:  {summary['total']:,}")
    print("="*60)       