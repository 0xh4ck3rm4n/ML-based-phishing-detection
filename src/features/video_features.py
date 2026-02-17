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
    def extract_temporal_features(self, frames: List[np.ndarray]) -> dict:
        features = {
            'motion_magnitude': 0,
            'optical_flow_mean': 0,
            'optical_flow_std': 0,
            'frame_difference_mean': 0
        }
        
        if len(frames) < 2:
            return features
        
        frame_diffs = []
        optical_flows = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(curr_gray, prev_gray)
            frame_diffs.append(np.mean(diff))
            
            # Optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Magnitude of optical flow
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            optical_flows.append(np.mean(magnitude))
        
        if frame_diffs:
            features['frame_difference_mean'] = np.mean(frame_diffs)
            features['frame_difference_std'] = np.std(frame_diffs)
        
        if optical_flows:
            features['optical_flow_mean'] = np.mean(optical_flows)
            features['optical_flow_std'] = np.std(optical_flows)
            features['motion_magnitude'] = np.max(optical_flows)
        
        return features
    
    # This function is responsible for extracting color-based features
    def extract_color_features(self, frames: List[np.ndarray]) -> dict:
        features = {}
        
        color_means = []
        color_stds = []
        
        for frame in frames:
            # Convert to different color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # BGR statistics
            color_means.append(np.mean(frame, axis=(0, 1)))
            color_stds.append(np.std(frame, axis=(0, 1)))
        
        color_means = np.array(color_means)
        color_stds = np.array(color_stds)
        
        features['color_mean_b'] = np.mean(color_means[:, 0])
        features['color_mean_g'] = np.mean(color_means[:, 1])
        features['color_mean_r'] = np.mean(color_means[:, 2])
        
        features['color_std_b'] = np.mean(color_stds[:, 0])
        features['color_std_g'] = np.mean(color_stds[:, 1])
        features['color_std_r'] = np.mean(color_stds[:, 2])
        
        # Color consistency across frames
        features['color_temporal_consistency'] = np.mean(np.std(color_means, axis=0))
        
        return features
    
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