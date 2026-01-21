
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Any


class FrameData:
    """
    Container for a single spectroscopy frame's data and analysis results.
    
    Attributes:
    -----------
    index : int
        Position in the frame sequence
    filename : str
        Original filename
    image_path : str
        Full path to image file
    image_data : np.ndarray
        Raw grayscale image data
    timestamp : float
        Time in microseconds relative to trigger
    
    ROI Properties:
    ---------------
    roi_offset_y : int
        Vertical offset for this frame's ROI (tracking lightning movement)
    
    Calibration Properties:
    -----------------------
    is_calibrated : bool
        Whether this frame has been manually calibrated
    calibration_points : list
        List of (pixel, wavelength) tuples for this frame
    calibration_coeffs : list
        Polynomial coefficients for this frame (if calibrated)
    pixel_offset : float
        Interpolated offset from reference frame
    
    Analysis Results:
    -----------------
    spectrum : np.ndarray
        Raw 1D spectrum array
    spectrum_baseline_removed : np.ndarray
        Processed spectrum for display
    peaks : list
        Detected peaks with positions and intensities
    wavelengths : np.ndarray
        Calibrated wavelength values
    assigned_lines : dict
        Peak-to-known-line assignments
    """
    
    def __init__(self, index: int, filename: str, image_path: str):
        self.index = index
        self.filename = filename
        self.image_path = image_path
        self.image_data: Optional[np.ndarray] = None
        self.timestamp: Optional[float] = None
        
        # ROI for this frame (can have per-frame y-offset)
        self.roi_offset_y = 0
        
        # Calibration - new multi-frame approach
        self.is_calibrated = False  # True if manually calibrated
        self.calibration_points: List[tuple] = []  # [(pixel, wavelength), ...]
        self.calibration_coeffs: Optional[List[float]] = None
        self.pixel_offset = 0.0  # Interpolated offset from reference
        
        # Legacy keyframe support (for backwards compatibility)
        self.is_keyframe = False
        self.keyframe_anchor_pixel: Optional[float] = None
        
        # Analysis results
        self.spectrum: Optional[np.ndarray] = None
        self.spectrum_baseline_removed: Optional[np.ndarray] = None
        self.peaks: Optional[List[Dict]] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.assigned_lines: Optional[Dict] = None
        
        # Ratio calculations (for quick access)
        self.ratio_656_777: Optional[float] = None
        self.ratio_463_656: Optional[float] = None
        
    def load_image(self) -> np.ndarray:
        """Load image data from file."""
        img = Image.open(self.image_path)
        self.image_data = np.array(img, dtype=np.float64)
        if len(self.image_data.shape) == 3:
            # Convert RGB to grayscale
            self.image_data = np.mean(self.image_data, axis=2)
        return self.image_data
    
    def unload_image(self):
        """Free memory by unloading image data."""
        self.image_data = None
    
    def get_effective_roi(self, base_roi: Dict) -> Dict:
        """
        Get the effective ROI for this frame, accounting for y-offset.
        
        Parameters:
        -----------
        base_roi : dict
            Base ROI with keys 'x', 'y', 'width', 'height'
            
        Returns:
        --------
        dict : ROI with y-offset applied
        """
        return {
            'x': base_roi['x'],
            'y': base_roi['y'] + self.roi_offset_y,
            'width': base_roi['width'],
            'height': base_roi['height']
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame data to dictionary for serialization."""
        return {
            'index': self.index,
            'filename': self.filename,
            'roi_offset_y': self.roi_offset_y,
            'is_calibrated': self.is_calibrated,
            'calibration_points': self.calibration_points,
            'calibration_coeffs': self.calibration_coeffs,
            'pixel_offset': self.pixel_offset,
            'is_keyframe': self.is_keyframe,
            'keyframe_anchor_pixel': self.keyframe_anchor_pixel,
            'ratio_656_777': self.ratio_656_777,
            'ratio_463_656': self.ratio_463_656,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], image_path: str) -> 'FrameData':
        """Create FrameData from dictionary."""
        frame = cls(
            index=data.get('index', 0),
            filename=data.get('filename', ''),
            image_path=image_path
        )
        frame.roi_offset_y = data.get('roi_offset_y', 0)
        frame.is_calibrated = data.get('is_calibrated', False)
        frame.calibration_points = data.get('calibration_points', [])
        frame.calibration_coeffs = data.get('calibration_coeffs')
        frame.pixel_offset = data.get('pixel_offset', 0.0)
        frame.is_keyframe = data.get('is_keyframe', False)
        frame.keyframe_anchor_pixel = data.get('keyframe_anchor_pixel')
        frame.ratio_656_777 = data.get('ratio_656_777')
        frame.ratio_463_656 = data.get('ratio_463_656')
        return frame
    
    def __repr__(self):
        cal_status = "calibrated" if self.is_calibrated else "uncalibrated"
        return f"FrameData(idx={self.index}, {self.filename}, {cal_status})"
