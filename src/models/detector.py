from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.url_features import URLFeatureExtractor
from features.text_features import TextFeatureExtractor, EmailFeatureExtractor, SMSFeatureExtractor
from features.audio_features import AudioFeatureExtractor, DeepfakeAudioDetector
from features.video_features import VideoFeatureExtractor, DeepfakeVideoDetector

class MultiModalDetector:
    # This function initializes multi-modal detector
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        use_transformers: bool = False
    ):
        self.url_extractor = URLFeatureExtractor()
        self.text_extractor = TextFeatureExtractor(use_transformers=use_transformers)
        self.email_extractor = EmailFeatureExtractor(use_transformers=use_transformers)
        self.sms_extractor = SMSFeatureExtractor(use_transformers=use_transformers)
        self.audio_extractor = AudioFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor()

        self.deepfake_audio_detector = DeepfakeAudioDetector()
        self.deepfake_video_detector = DeepfakeVideoDetector()

        # models
        self.url_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.email_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.sms_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.audio_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.video_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_trained = {
            'url': False,
            'email': False,
            'sms': False,
            'audio': False,
            'video': False
        }