import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os

class VideoFeatureExtractor:
    
    def __init__(self, target_fps: int = 5, max_frames: int = 100):
        self.target_fps = target_fps
        self.max_frames = max_frames

    # This function is responsible for loading video and extract frames
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to match target FPS
        frame_skip = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        return frames, fps

    # This function extract face-based features using Haar Cascade
    def extract_face_features(self, frames: List[np.ndarray]) -> dict:
        """
        Args:
            frames: list of video frames

        Returns:
            Dictionary of face-related features
        """
        pass
    
    # This function extract temporal features such as motion, optical flow
    def extract_temporal_features():
        pass
    
    # This function is responsible for extracting color-based features
    def extract_color_features():
        pass
    
    # This functione extract texture features using edge detection
    def extract_texture_features(self, frames: List[np.ndarray]) -> dict:
        features = {}
        
        edge_densities = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
        
        features['edge_density_mean'] = np.mean(edge_densities)
        features['edge_density_std'] = np.std(edge_densities)
        
        return features
    
    # This function extract frequency domain features (FFT-based)
    def extract_frequency_features():
        pass

    def extract_all_features():
        pass

    def extract_batch():
        pass

    def get_features_name():
        pass