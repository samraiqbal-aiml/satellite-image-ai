#!/usr/bin/env python3
"""
Satellite Image Quality Assessment
Author: Samra Iqbal
Description: Assess quality metrics of satellite imagery
"""

import cv2
import numpy as np
from scipy import ndimage

class ImageQualityAssessor:
    def __init__(self):
        print("🎯 Image Quality Assessor Initialized")
    
    def assess_quality(self, image):
        """Comprehensive quality assessment of satellite image"""
        if image is None:
            return self.default_quality_report()
        
        # Calculate various quality metrics
        sharpness = self.calculate_sharpness(image)
        contrast = self.calculate_contrast(image)
        noise_level = self.calculate_noise_level(image)
        brightness = self.calculate_brightness(image)
        
        # Overall quality score (weighted average)
        overall_quality = (
            sharpness * 0.3 +
            contrast * 0.25 +
            (1 - noise_level) * 0.25 +
            brightness * 0.2
        )
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'noise_level': noise_level,
            'brightness': brightness,
            'overall_quality': overall_quality,
            'quality_level': self.get_quality_level(overall_quality)
        }
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() / 1000  # Normalized
    
    def calculate_contrast(self, image):
        """Calculate image contrast using standard deviation"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray) / 128  # Normalized to 0-1 range
    
    def calculate_noise_level(self, image):
        """Estimate noise level using median absolute deviation"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        return min(mad / 64, 1.0)  # Normalized
    
    def calculate_brightness(self, image):
        """Calculate normalized brightness level"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray) / 255  # Normalize to 0-1
        # Ideal brightness is around 0.5, penalize extremes
        return 1 - abs(brightness - 0.5) * 2
    
    def get_quality_level(self, score):
        """Convert numerical score to quality level"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def default_quality_report(self):
        """Return default report when no image is available"""
        return {
            'sharpness': 0.5,
            'contrast': 0.5,
            'noise_level': 0.3,
            'brightness': 0.7,
            'overall_quality': 0.5,
            'quality_level': "Estimated"
        }
