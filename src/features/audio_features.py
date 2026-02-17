import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")
    
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc
        )
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        return features
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}

        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_voiced = f0[f0 > librosa.note_to_hz('C2')]
        
        if len(f0_voiced) > 0:
            features['pitch_mean'] = np.mean(f0_voiced)
            features['pitch_std'] = np.std(f0_voiced)
            features['pitch_min'] = np.min(f0_voiced)
            features['pitch_max'] = np.max(f0_voiced)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0
        
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        features['duration'] = len(audio) / sr
        
        return features
    
    def extract_chroma_features(self, audio: np.ndarray, sr: int) -> dict:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        features = {}
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def extract_all_features(self, audio_path: str) -> np.ndarray:

        audio, sr = self.load_audio(audio_path)
        
        mfcc = self.extract_mfcc(audio, sr)
        mfcc_features = {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1)
        }

        mfcc_vector = np.concatenate([
            mfcc_features['mfcc_mean'],
            mfcc_features['mfcc_std']
        ])

        spectral_features = self.extract_spectral_features(audio, sr)
        prosodic_features = self.extract_prosodic_features(audio, sr)
        chroma_features = self.extract_chroma_features(audio, sr)

        feature_vector = np.concatenate([
            mfcc_vector,
            list(spectral_features.values()),
            list(prosodic_features.values()),
            list(chroma_features.values())
        ])
        
        return feature_vector
    
    def extract_batch(self, audio_paths: List[str]) -> np.ndarray:
        features = []
        
        for audio_path in audio_paths:
            try:
                feature_vector = self.extract_all_features(audio_path)
                features.append(feature_vector)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Add zero vector for failed files
                if len(features) > 0:
                    features.append(np.zeros_like(features[0]))
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        feature_names = []
        
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_{i}_mean')
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_{i}_std')

        feature_names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'zcr_mean', 'zcr_std',
            'spectral_contrast_mean', 'spectral_contrast_std'
        ])

        feature_names.extend([
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
            'rms_mean', 'rms_std', 'duration'
        ])

        feature_names.extend(['chroma_mean', 'chroma_std'])
        
        return feature_names


class DeepfakeAudioDetector:
    def __init__(self):
        self.extractor = AudioFeatureExtractor()
    
    def extract_deepfake_features(self, audio_path: str) -> dict:
        audio, sr = self.extractor.load_audio(audio_path)
        
        features = {}

        freqs = librosa.fft_frequencies(sr=sr)
        stft = np.abs(librosa.stft(audio))
        
        high_freq_idx = freqs > 4000  # Above 4kHz
        high_freq_energy = np.mean(stft[high_freq_idx, :])
        low_freq_energy = np.mean(stft[~high_freq_idx, :])
        
        features['high_freq_ratio'] = high_freq_energy / (low_freq_energy + 1e-10)
        
        phase = np.angle(librosa.stft(audio))
        phase_diff = np.diff(phase, axis=1)
        features['phase_consistency'] = np.std(phase_diff)
        
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
        
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_ratio'] = np.sum(harmonic**2) / (np.sum(audio**2) + 1e-10)
        
        return features


if __name__ == "__main__":
    # Test the extractor
    print("AudioFeatureExtractor - Ready to process audio files")
    print("Use extract_all_features(audio_path) to extract features from audio")
    
    extractor = AudioFeatureExtractor()
    print(f"Total features extracted: {len(extractor.get_feature_names())}")