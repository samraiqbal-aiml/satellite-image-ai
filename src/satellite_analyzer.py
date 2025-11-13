#!/usr/bin/env python3
"""
Satellite Image Analyzer
Author: Samra Iqbal
Description: Main module for satellite image processing and analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from object_detector import SatelliteObjectDetector
from quality_assessment import ImageQualityAssessor

class SatelliteAnalyzer:
    def __init__(self):
        self.object_detector = SatelliteObjectDetector()
        self.quality_assessor = ImageQualityAssessor()
        print("🛰️ Satellite Image Analyzer Initialized")
    
    def analyze_image(self, image_path):
        """Main analysis pipeline for satellite images"""
        print(f"🔍 Analyzing satellite image: {image_path}")
        
        # Load and preprocess image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Perform quality assessment
        quality_report = self.quality_assessor.assess_quality(image)
        
        # Detect objects
        detection_results = self.object_detector.detect_objects(image)
        
        # Generate comprehensive report
        analysis_report = {
            'image_info': {
                'dimensions': image.shape,
                'file_path': image_path
            },
            'quality_assessment': quality_report,
            'object_detection': detection_results,
            'overall_score': self.calculate_overall_score(quality_report, detection_results)
        }
        
        return analysis_report
    
    def load_image(self, image_path):
        """Load and preprocess satellite image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try creating a sample image for demo
                print("📸 Creating sample satellite image for demonstration...")
                image = self.create_sample_image()
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return self.create_sample_image()
    
    def create_sample_image(self):
        """Create a sample satellite image for demonstration"""
        # Create a realistic-looking satellite image
        height, width = 512, 512
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add blue water bodies
        cv2.rectangle(image, (100, 100), (200, 200), (70, 130, 180), -1)  # Water
        cv2.rectangle(image, (300, 300), (400, 400), (80, 140, 190), -1)  # Water
        
        # Add green vegetation
        cv2.rectangle(image, (150, 350), (250, 450), (40, 180, 40), -1)   # Forest
        cv2.rectangle(image, (350, 150), (450, 250), (50, 160, 50), -1)   # Forest
        
        # Add urban areas (gray)
        cv2.rectangle(image, (50, 50), (120, 120), (120, 120, 120), -1)   # Urban
        cv2.rectangle(image, (400, 400), (480, 480), (100, 100, 100), -1) # Urban
        
        # Add roads (white lines)
        cv2.line(image, (0, 256), (512, 256), (200, 200, 200), 3)
        cv2.line(image, (256, 0), (256, 512), (200, 200, 200), 3)
        
        return image
    
    def calculate_overall_score(self, quality_report, detection_results):
        """Calculate overall image quality and usefulness score"""
        quality_score = quality_report.get('overall_quality', 0)
        detection_confidence = detection_results.get('confidence', 0)
        
        overall_score = (quality_score * 0.6) + (detection_confidence * 0.4)
        return min(overall_score, 1.0)
    
    def generate_report(self, analysis_report):
        """Generate a printable analysis report"""
        if not analysis_report:
            return "No analysis data available"
        
        report = []
        report.append("=" * 50)
        report.append("🛰️ SATELLITE IMAGE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Image info
        info = analysis_report['image_info']
        report.append(f"📊 Image Dimensions: {info['dimensions']}")
        report.append(f"📁 File: {info['file_path']}")
        
        # Quality assessment
        quality = analysis_report['quality_assessment']
        report.append("\n🎯 QUALITY ASSESSMENT:")
        report.append(f"   - Overall Quality: {quality['overall_quality']:.2f}/1.0")
        report.append(f"   - Sharpness: {quality['sharpness']:.2f}")
        report.append(f"   - Contrast: {quality['contrast']:.2f}")
        report.append(f"   - Noise Level: {quality['noise_level']:.2f}")
        
        # Object detection
        detection = analysis_report['object_detection']
        report.append("\n🔍 OBJECT DETECTION:")
        report.append(f"   - Objects Found: {len(detection['objects'])}")
        report.append(f"   - Detection Confidence: {detection['confidence']:.2f}")
        
        for obj in detection['objects']:
            report.append(f"     • {obj['type']} (Confidence: {obj['confidence']:.2f})")
        
        # Overall score
        report.append(f"\n⭐ OVERALL USEFULNESS SCORE: {analysis_report['overall_score']:.2f}/1.0")
        report.append("=" * 50)
        
        return "\n".join(report)

def main():
    """Main demonstration function"""
    analyzer = SatelliteAnalyzer()
    
    # Analyze a sample image
    analysis = analyzer.analyze_image("sample_satellite.jpg")
    
    if analysis:
        report = analyzer.generate_report(analysis)
        print(report)
        
        # Save sample output
        with open("analysis_report.txt", "w") as f:
            f.write(report)
        print("💾 Analysis report saved to 'analysis_report.txt'")

if __name__ == "__main__":
    main()
