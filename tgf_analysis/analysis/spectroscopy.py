import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import ndimage
from scipy.signal import find_peaks

# Try to import optional dependencies
try:
    import peakutils
    HAS_PEAKUTILS = True
except ImportError:
    HAS_PEAKUTILS = False

from config import KNOWN_LINES


class SpectroscopyAnalyzer:
    """
    Static methods for spectroscopic analysis.
    
    Provides:
    - Spectrum extraction from images
    - Baseline removal
    - Peak detection and refinement
    - Wavelength calibration
    - Line identification
    - Ratio calculations
    """
    
    @staticmethod
    def extract_spectrum(image: np.ndarray, roi: Dict, 
                        tilt_angle: float = 0.0,
                        flip_horizontal: bool = False) -> np.ndarray:
        """
        Extract 1D spectrum from image ROI.
        
        Parameters:
        -----------
        image : np.ndarray
            2D grayscale image
        roi : dict
            ROI with keys 'x', 'y', 'width', 'height'
        tilt_angle : float
            Tilt correction angle in degrees
        flip_horizontal : bool
            Whether to flip the spectrum
            
        Returns:
        --------
        np.ndarray : 1D spectrum array
        """
        x, y = roi['x'], roi['y']
        w, h = roi['width'], roi['height']
        
        # Ensure ROI is within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - w))
        y = max(0, min(y, img_h - h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Extract ROI
        roi_data = image[y:y+h, x:x+w].copy()
        
        # Apply tilt correction if needed
        if abs(tilt_angle) > 0.01:
            roi_data = ndimage.rotate(roi_data, tilt_angle, 
                                     reshape=False, mode='nearest')
        
        # Average vertically to get 1D spectrum
        spectrum = np.mean(roi_data, axis=0)
        
        # Flip if needed
        if flip_horizontal:
            spectrum = spectrum[::-1]
        
        return spectrum
    
    @staticmethod
    def remove_baseline(spectrum: np.ndarray, degree: int = 8) -> np.ndarray:
        """
        Remove baseline from spectrum using iterative polynomial fit.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        degree : int
            Polynomial degree for baseline fit
            
        Returns:
        --------
        np.ndarray : Baseline-removed spectrum
        """
        if HAS_PEAKUTILS:
            try:
                baseline = peakutils.baseline(spectrum, deg=degree)
                return spectrum - baseline
            except:
                pass
        
        # Fallback method
        return SpectroscopyAnalyzer._remove_baseline_fallback(spectrum, degree)
    
    @staticmethod
    def _remove_baseline_fallback(spectrum: np.ndarray, degree: int = 8) -> np.ndarray:
        """Fallback baseline removal without peakutils."""
        x = np.arange(len(spectrum))
        
        # Iterative fitting excluding peaks
        mask = np.ones(len(spectrum), dtype=bool)
        
        for _ in range(3):
            if np.sum(mask) < degree + 1:
                break
            
            coeffs = np.polyfit(x[mask], spectrum[mask], degree)
            baseline = np.polyval(coeffs, x)
            residual = spectrum - baseline
            
            # Update mask to exclude positive outliers (peaks)
            threshold = 0.5 * np.std(residual[mask])
            mask = residual < threshold
        
        return spectrum - baseline
    
    @staticmethod
    def find_peaks_in_spectrum(spectrum: np.ndarray,
                               prominence: float = 0.15,
                               distance: int = 10,
                               height: float = 0.1) -> List[Dict]:
        """
        Find peaks in spectrum.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum (should be baseline-removed)
        prominence : float
            Minimum peak prominence (fraction of max)
        distance : int
            Minimum distance between peaks in pixels
        height : float
            Minimum peak height (fraction of max)
            
        Returns:
        --------
        list : List of peak dictionaries with 'position', 'intensity', 'prominence'
        """
        # Normalize
        spec_max = np.max(spectrum)
        if spec_max <= 0:
            return []
        
        spec_norm = spectrum / spec_max
        
        # Find peaks
        peaks, properties = find_peaks(
            spec_norm,
            prominence=prominence,
            distance=distance,
            height=height
        )
        
        # Build results
        results = []
        for i, peak_idx in enumerate(peaks):
            results.append({
                'position': float(peak_idx),
                'intensity': float(spectrum[peak_idx]),
                'intensity_normalized': float(spec_norm[peak_idx]),
                'prominence': float(properties['prominences'][i]) if 'prominences' in properties else 0
            })
        
        return results
    
    @staticmethod
    def refine_peak_position(spectrum: np.ndarray, peak_idx: int, 
                            window: int = 3) -> float:
        """
        Refine peak position using quadratic interpolation.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Input spectrum
        peak_idx : int
            Initial peak index
        window : int
            Half-window size for fitting
            
        Returns:
        --------
        float : Refined peak position
        """
        left = max(0, peak_idx - window)
        right = min(len(spectrum), peak_idx + window + 1)
        
        if right - left < 3:
            return float(peak_idx)
        
        x = np.arange(left, right)
        y = spectrum[left:right]
        
        try:
            coeffs = np.polyfit(x, y, 2)
            if coeffs[0] < 0:  # Concave down (valid peak)
                refined = -coeffs[1] / (2 * coeffs[0])
                if left <= refined <= right:
                    return refined
        except:
            pass
        
        return float(peak_idx)
    
    @staticmethod
    def polynomial_fit(calibration_points: List[Tuple[float, float]],
                      order: int = 4) -> Tuple[List[float], float]:
        """
        Fit polynomial for wavelength calibration.
        
        Parameters:
        -----------
        calibration_points : list
            List of (pixel, wavelength) tuples
        order : int
            Polynomial order
            
        Returns:
        --------
        tuple : (coefficients, r_squared)
        """
        if len(calibration_points) < order + 1:
            order = len(calibration_points) - 1
        
        if order < 1:
            return None, 0.0
        
        pixels = np.array([p[0] for p in calibration_points])
        wavelengths = np.array([p[1] for p in calibration_points])
        
        try:
            coeffs = np.polyfit(pixels, wavelengths, order)
            
            # Calculate R-squared
            predicted = np.polyval(coeffs, pixels)
            ss_res = np.sum((wavelengths - predicted)**2)
            ss_tot = np.sum((wavelengths - np.mean(wavelengths))**2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return coeffs.tolist(), r_squared
            
        except Exception as e:
            print(f"Error in polynomial fit: {e}")
            return None, 0.0
    
    @staticmethod
    def pixel_to_wavelength(pixels: np.ndarray, 
                           coeffs: List[float]) -> np.ndarray:
        """Convert pixel positions to wavelengths using calibration."""
        return np.polyval(coeffs, pixels)
    
    @staticmethod
    def wavelength_to_pixel(wavelengths: np.ndarray,
                           coeffs: List[float]) -> np.ndarray:
        """
        Convert wavelengths to approximate pixel positions.
        
        Uses numerical root finding since polynomial is pixel->wavelength.
        """
        # Create inverse lookup
        pixels = np.arange(1000)
        wls = np.polyval(coeffs, pixels)
        
        result = np.interp(wavelengths, wls, pixels)
        return result
    
    @staticmethod
    def identify_lines(peaks: List[Dict], coeffs: List[float],
                      tolerance: float = 5.0) -> List[Dict]:
        """
        Identify peaks with known spectral lines.
        
        Parameters:
        -----------
        peaks : list
            List of peak dictionaries
        coeffs : list
            Calibration coefficients
        tolerance : float
            Wavelength tolerance in nm
            
        Returns:
        --------
        list : Peaks with added 'wavelength' and 'identification' fields
        """
        for peak in peaks:
            wl = np.polyval(coeffs, peak['position'])
            peak['wavelength'] = wl
            peak['identification'] = None
            peak['species'] = None
            
            # Try to match with known lines
            for known_wl, (species, priority, is_anchor) in KNOWN_LINES.items():
                if abs(wl - known_wl) < tolerance:
                    peak['identification'] = known_wl
                    peak['species'] = species
                    break
        
        return peaks
    
    @staticmethod
    def calculate_ratios(wavelengths: np.ndarray, spectrum: np.ndarray,
                        tolerance: float = 10.0) -> Dict:
        """
        Calculate important spectral ratios.
        
        Returns ratios for:
        - 656/777 (Hα/OI) - ionization/temperature indicator
        - 463/656 (NII/Hα) - nitrogen ionization
        - 500/656 (NII/Hα) - alternative nitrogen
        """
        def find_peak_intensity(target_wl):
            mask = np.abs(wavelengths - target_wl) <= tolerance
            if not np.any(mask):
                return None
            return np.max(spectrum[mask])
        
        ratios = {}
        
        # Get intensities at key wavelengths
        i_656 = find_peak_intensity(656)
        i_777 = find_peak_intensity(777)
        i_463 = find_peak_intensity(463)
        i_500 = find_peak_intensity(500)
        
        # Calculate ratios
        if i_656 is not None and i_777 is not None and i_777 > 0:
            ratios['656_777'] = i_656 / i_777
        
        if i_463 is not None and i_656 is not None and i_656 > 0:
            ratios['463_656'] = i_463 / i_656
        
        if i_500 is not None and i_656 is not None and i_656 > 0:
            ratios['500_656'] = i_500 / i_656
        
        return ratios
    
    @staticmethod
    def estimate_temperature(ratio_656_777: float) -> float:
        """
        Estimate plasma temperature from 656/777 ratio.
        
        This is a simplified estimate based on empirical relationships.
        Real temperature calculation requires more sophisticated analysis.
        
        Returns temperature in Kelvin.
        """
        # Simplified linear relationship
        # Based on literature values for lightning channels
        T = 8000 + ratio_656_777 * 6000
        return max(5000, min(35000, T))
    
    @staticmethod
    def interpolate_calibration(frame_idx: int,
                               calibrated_frames: Dict[int, Dict]) -> Optional[List[float]]:
        """
        Interpolate calibration coefficients between calibrated frames.
        
        Parameters:
        -----------
        frame_idx : int
            Frame index to interpolate for
        calibrated_frames : dict
            {frame_idx: {'coeffs': [...], 'points': [...]}}
            
        Returns:
        --------
        list : Interpolated coefficients
        """
        if not calibrated_frames:
            return None
        
        # Get sorted frame indices
        indices = sorted(calibrated_frames.keys())
        
        if len(indices) == 1:
            # Only one calibrated frame - use its coefficients
            return calibrated_frames[indices[0]]['coeffs']
        
        # Find bracketing frames
        if frame_idx <= indices[0]:
            return calibrated_frames[indices[0]]['coeffs']
        if frame_idx >= indices[-1]:
            return calibrated_frames[indices[-1]]['coeffs']
        
        # Find the two frames to interpolate between
        for i, idx in enumerate(indices[:-1]):
            if idx <= frame_idx < indices[i+1]:
                idx1, idx2 = idx, indices[i+1]
                break
        
        # Linear interpolation of coefficients
        coeffs1 = np.array(calibrated_frames[idx1]['coeffs'])
        coeffs2 = np.array(calibrated_frames[idx2]['coeffs'])
        
        # Interpolation factor
        t = (frame_idx - idx1) / (idx2 - idx1)
        
        interpolated = coeffs1 * (1 - t) + coeffs2 * t
        
        return interpolated.tolist()
    
    @staticmethod
    def suggest_roi_offset(current_image: np.ndarray,
                          reference_spectrum: np.ndarray,
                          base_roi: Dict,
                          search_range: int = 20) -> Tuple[int, float]:
        """
        Suggest vertical ROI offset for tracking lightning movement.
        
        Parameters:
        -----------
        current_image : np.ndarray
            Current frame image
        reference_spectrum : np.ndarray
            Reference spectrum to match
        base_roi : dict
            Base ROI definition
        search_range : int
            Pixels to search above and below
            
        Returns:
        --------
        tuple : (suggested_offset, confidence)
        """
        best_offset = 0
        best_score = -1
        
        for offset in range(-search_range, search_range + 1):
            test_roi = base_roi.copy()
            test_roi['y'] = base_roi['y'] + offset
            
            # Check bounds
            if test_roi['y'] < 0:
                continue
            if test_roi['y'] + test_roi['height'] > current_image.shape[0]:
                continue
            
            # Extract spectrum
            spectrum = SpectroscopyAnalyzer.extract_spectrum(current_image, test_roi)
            
            # Score by correlation with reference
            if len(spectrum) == len(reference_spectrum):
                corr = np.corrcoef(spectrum, reference_spectrum)[0, 1]
                if corr > best_score:
                    best_score = corr
                    best_offset = offset
        
        return best_offset, best_score
