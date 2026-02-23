"""
TGF Analysis Tool - Luminosity Data Handler
==========================================
Handles loading and processing of high-speed camera luminosity data.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image


class LuminosityHandler:
    """
    Handles luminosity data from high-speed cameras.
    
    Can either:
    1. Load pre-computed luminosity text files
    2. Calculate luminosity from image sequences
    """
    
    def __init__(self):
        """Initialize the LuminosityHandler."""
        self.filename: Optional[str] = None
        self.filepath: Optional[str] = None
        
        # Loaded data
        self.time: Optional[np.ndarray] = None  # Time in µs
        self.luminosity: Optional[np.ndarray] = None  # Raw luminosity values
        self.luminosity_normalized: Optional[np.ndarray] = None  # 0-1 normalized
        
        # Image sequence parameters (for calculating from images)
        self.image_directory: Optional[str] = None
        self.frame_interval: float = 25.0  # µs between frames
        self.start_time: float = 0.0  # Start time in µs
        self.roi: Dict = {'x': 800, 'y': 280, 'width': 100, 'height': 140}
        
        # Background subtraction
        self.background: Optional[np.ndarray] = None
        
        # Data loaded flag
        self.is_loaded = False
    
    def load_text_file(self, filepath: str) -> bool:
        """
        Load pre-computed luminosity data from text file.
        
        Expected format: Time(µs) Luminosity
        """
        try:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            
            data = np.loadtxt(filepath)
            
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            self.time = data[:, 0]
            self.luminosity = data[:, 1]
            
            # Normalize
            max_val = np.max(self.luminosity)
            if max_val > 0:
                self.luminosity_normalized = self.luminosity / max_val
            else:
                self.luminosity_normalized = self.luminosity.copy()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading luminosity file: {e}")
            self.is_loaded = False
            return False
    
    def set_roi(self, x: int, y: int, width: int, height: int):
        """Set the region of interest for luminosity calculation."""
        self.roi = {'x': x, 'y': y, 'width': width, 'height': height}
    
    def set_timing(self, start_time: float, frame_interval: float):
        """Set timing parameters for image sequence."""
        self.start_time = start_time
        self.frame_interval = frame_interval
    
    def calculate_background(self, image_paths: List[str]) -> np.ndarray:
        """
        Calculate background from a list of images.
        
        Parameters:
        -----------
        image_paths : list
            List of paths to background images
            
        Returns:
        --------
        np.ndarray : Mean background image
        """
        if not image_paths:
            return None
        
        backgrounds = []
        for path in image_paths:
            try:
                img = np.array(Image.open(path), dtype=np.float64)
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)
                backgrounds.append(img)
            except Exception as e:
                print(f"Error loading background image {path}: {e}")
        
        if backgrounds:
            self.background = np.median(backgrounds, axis=0)
            return self.background
        return None
    
    def calculate_luminosity_from_images(self, image_directory: str,
                                         start_frame: int = 0,
                                         end_frame: int = None,
                                         bg_frames: int = 10) -> bool:
        """
        Calculate luminosity from an image sequence.
        
        Parameters:
        -----------
        image_directory : str
            Path to directory containing .tif images
        start_frame : int
            First frame to process
        end_frame : int
            Last frame to process (None for all)
        bg_frames : int
            Number of frames to use for background
        """
        try:
            self.image_directory = image_directory
            
            # Get list of image files
            image_files = sorted([
                f for f in os.listdir(image_directory)
                if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
            ])
            
            if not image_files:
                print("No image files found")
                return False
            
            # Calculate background from first N frames
            bg_paths = [
                os.path.join(image_directory, image_files[i])
                for i in range(min(bg_frames, len(image_files)))
            ]
            self.calculate_background(bg_paths)
            
            # Process frames
            times = []
            luminosities = []
            
            if end_frame is None:
                end_frame = len(image_files)
            
            for i, filename in enumerate(image_files[start_frame:end_frame]):
                frame_num = start_frame + i
                
                # Load image
                img_path = os.path.join(image_directory, filename)
                img = np.array(Image.open(img_path), dtype=np.float64)
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)
                
                # Subtract background
                if self.background is not None:
                    img = img - self.background
                
                # Extract ROI
                x, y = self.roi['x'], self.roi['y']
                w, h = self.roi['width'], self.roi['height']
                roi_data = img[y:y+h, x:x+w]
                
                # Calculate luminosity (sum of pixel values)
                lum = np.sum(roi_data)
                
                # Calculate time
                t = self.start_time + frame_num * self.frame_interval
                
                times.append(t)
                luminosities.append(lum)
            
            self.time = np.array(times)
            self.luminosity = np.array(luminosities)
            
            # Normalize
            max_val = np.max(self.luminosity)
            if max_val > 0:
                self.luminosity_normalized = self.luminosity / max_val
            else:
                self.luminosity_normalized = self.luminosity.copy()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error calculating luminosity: {e}")
            self.is_loaded = False
            return False
    
    def save_to_file(self, output_filepath: str) -> bool:
        """Save luminosity data to text file."""
        if not self.is_loaded:
            return False
        
        try:
            with open(output_filepath, 'w') as f:
                f.write("#Time Lum\n")
                for t, lum in zip(self.time, self.luminosity):
                    f.write(f"{t:.6f} {lum:.6f}\n")
            return True
        except Exception as e:
            print(f"Error saving luminosity file: {e}")
            return False
    
    def get_data(self, normalized: bool = True) -> Dict:
        """
        Get luminosity data.
        
        Parameters:
        -----------
        normalized : bool
            If True, return normalized luminosity (0-1)
        """
        if not self.is_loaded:
            return None
        
        return {
            'time': self.time,
            'luminosity': self.luminosity_normalized if normalized else self.luminosity
        }
    
    def get_data_in_range(self, t_min: float, t_max: float, 
                         normalized: bool = True) -> Dict:
        """Get luminosity data within a time range."""
        if not self.is_loaded:
            return None
        
        mask = (self.time >= t_min) & (self.time <= t_max)
        lum = self.luminosity_normalized if normalized else self.luminosity
        
        return {
            'time': self.time[mask],
            'luminosity': lum[mask]
        }
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range of the data."""
        if not self.is_loaded:
            return None, None
        return self.time[0], self.time[-1]
    
    def find_return_stroke(self, threshold: float = 0.9) -> Optional[float]:
        """
        Find the return stroke time based on luminosity peak.
        
        Parameters:
        -----------
        threshold : float
            Fraction of max luminosity to identify as RS
        """
        if not self.is_loaded:
            return None
        
        max_idx = np.argmax(self.luminosity)
        if self.luminosity_normalized[max_idx] >= threshold:
            return self.time[max_idx]
        return None
    
    def get_info(self) -> str:
        """Get summary information about loaded data."""
        if not self.is_loaded:
            return "No data loaded"
        
        t_min, t_max = self.get_time_range()
        
        return (
            f"File: {self.filename or 'Calculated from images'}\n"
            f"Points: {len(self.time):,}\n"
            f"Time Range: {t_min:.0f} - {t_max:.0f} µs\n"
            f"ROI: ({self.roi['x']}, {self.roi['y']}) {self.roi['width']}x{self.roi['height']}\n"
            f"Frame Interval: {self.frame_interval} µs"
        )
