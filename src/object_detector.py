#!/usr/bin/env python3
"""
Satellite Object Detector
Author: Samra Iqbal
Description: Detect objects in satellite imagery using computer vision
"""

import cv2
import numpy as np

class SatelliteObjectDetector:
    def __init__(self):
        print("🔍 Satellite Object Detector Initialized")
    
    def detect_objects(self, image):
        """Detect common objects in satellite imagery"""
        objects_detected = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect water bodies (blue regions)
        water_objects = self.detect_water(hsv)
        objects_detected.extend(water_objects)
        
        # Detect vegetation (green regions)
        vegetation_objects = self.detect_vegetation(hsv)
        objects_detected.extend(vegetation_objects)
        
        # Detect urban areas (gray regions)
        urban_objects = self.detect_urban_areas(image)
        objects_detected.extend(urban_objects)
        
        # Calculate overall confidence
        confidence = self.calculate_detection_confidence(objects_detected)
        
        return {
            'objects': objects_detected,
            'confidence': confidence,
            'total_objects': len(objects_detected)
        }
    
    def detect_water(self, hsv_image):
        """Detect water bodies using color thresholding"""
        # Define blue color range for water
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        water_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small detections
                water_objects.append({
                    'type': 'water_body',
                    'confidence': min(area / 10000, 1.0),
                    'area': area
                })
        
        return water_objects
    
    def detect_vegetation(self, hsv_image):
        """Detect vegetation using green color thresholding"""
        # Define green color range for vegetation
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vegetation_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                vegetation_objects.append({
                    'type': 'vegetation',
                    'confidence': min(area / 8000, 1.0),
                    'area': area
                })
        
        return vegetation_objects
    
    def detect_urban_areas(self, image):
        """Detect urban areas using edge detection and morphology"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations to find structured areas
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        urban_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                urban_objects.append({
                    'type': 'urban_area',
                    'confidence': min(area / 15000, 1.0),
                    'area': area
                })
        
        return urban_objects
    
    def calculate_detection_confidence(self, objects):
        """Calculate overall detection confidence"""
        if not objects:
            return 0.0
        
        total_confidence = sum(obj['confidence'] for obj in objects)
        return min(total_confidence / len(objects), 1.0)
