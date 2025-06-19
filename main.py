import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
import time
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


class YOLODetector:
    """Minimal YOLO-style detector for real-time inference"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        # Simulate trained model weights
        self.model_loaded = True
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        Returns: List of detections with bbox, confidence, class
        """
        # Simulate YOLO detection with dummy results for demo
        h, w = frame.shape[:2]
        
        # Simulate some detections
        detections = []
        
        # Add some realistic dummy detections
        if np.random.random() > 0.3:  # 70% chance of detection
            for _ in range(np.random.randint(1, 4)):
                x = np.random.randint(0, w-100)
                y = np.random.randint(0, h-100)
                w_box = np.random.randint(50, 150)
                h_box = np.random.randint(50, 150)
                conf = np.random.uniform(0.6, 0.95)
                
                detections.append({
                    'bbox': [x, y, w_box, h_box],
                    'confidence': conf,
                    'class': 'incident'
                })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        result = frame.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{det['class']}: {conf:.2f}"
            cv2.putText(result, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result


class ResNetSeverityClassifier:
    """ResNet18-based severity classifier"""
    
    def __init__(self, num_classes: int = 3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(num_classes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.severity_labels = ['Low', 'Medium', 'High']
        
    def _build_model(self, num_classes: int) -> nn.Module:
        """Build ResNet18 model"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(self.device)
        model.eval()
        return model
    
    def classify_severity(self, cropped_incident: np.ndarray) -> Tuple[str, float]:
        """
        Classify severity of cropped incident
        Returns: (severity_label, confidence)
        """
        # Preprocess image
        if len(cropped_incident.shape) == 3:
            input_tensor = self.transform(cropped_incident).unsqueeze(0).to(self.device)
        else:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(cropped_incident, cv2.COLOR_GRAY2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        return self.severity_labels[predicted.item()], confidence.item()


class ThermalProcessor:
    """AMG8833 thermal data processor with fire detection"""
    
    def __init__(self, fire_threshold: float = 150.0):
        self.fire_threshold = fire_threshold
        self.grid_size = (8, 8)  # AMG8833 resolution
        
    def simulate_thermal_data(self) -> np.ndarray:
        """Simulate AMG8833 thermal sensor data"""
        # Base temperature around 25°C with some variation
        base_temp = 25.0
        thermal_data = np.random.normal(base_temp, 5.0, self.grid_size)
        
        # Occasionally add hot spots (fire simulation)
        if np.random.random() > 0.7:  # 30% chance of fire
            fire_x = np.random.randint(0, 8)
            fire_y = np.random.randint(0, 8)
            # Create fire hotspot
            thermal_data[fire_y, fire_x] = np.random.uniform(150, 200)
            # Add some spread
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = fire_x + dx, fire_y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        thermal_data[ny, nx] += np.random.uniform(20, 50)
        
        return thermal_data
    
    def detect_fire(self, thermal_data: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Detect fire using temperature threshold and clustering
        Returns: (fire_detected, fire_locations)
        """
        # Apply noise suppression (simple median filter)
        filtered_data = np.copy(thermal_data)
        
        # Find pixels above threshold
        fire_mask = thermal_data > self.fire_threshold
        fire_points = np.argwhere(fire_mask)
        
        if len(fire_points) == 0:
            return False, []
        
        # Use DBSCAN clustering to group fire pixels
        if len(fire_points) > 1:
            clustering = DBSCAN(eps=1.5, min_samples=1).fit(fire_points)
            labels = clustering.labels_
            
            # Get cluster centers
            fire_locations = []
            for label in set(labels):
                if label != -1:  # Not noise
                    cluster_points = fire_points[labels == label]
                    center = np.mean(cluster_points, axis=0).astype(int)
                    fire_locations.append(tuple(center))
        else:
            fire_locations = [tuple(fire_points[0])]
        
        return True, fire_locations
    
    def visualize_thermal(self, thermal_data: np.ndarray, fire_locations: List[Tuple[int, int]] = None):
        """Visualize thermal data with fire locations"""
        plt.figure(figsize=(8, 6))
        
        # Create thermal heatmap
        im = plt.imshow(thermal_data, cmap='hot', interpolation='nearest')
        plt.colorbar(im, label='Temperature (°C)')
        
        # Mark fire locations
        if fire_locations:
            for y, x in fire_locations:
                plt.plot(x, y, 'bx', markersize=15, markeredgewidth=3, label='Fire Detected')
        
        plt.title('AMG8833 Thermal Data')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        if fire_locations:
            plt.legend()
        
        plt.tight_layout()
        plt.show()


class IntegratedDetectionSystem:
    """Main system integrating all components"""
    
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.severity_classifier = ResNetSeverityClassifier()
        self.thermal_processor = ThermalProcessor()
        self.fps_counter = 0
        self.start_time = time.time()
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame through complete pipeline"""
        results = {
            'detections': [],
            'thermal_fire': False,
            'fire_locations': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # 1. YOLO Detection
        detections = self.yolo_detector.detect(frame)
        
        # 2. Severity Classification for each detection
        for detection in detections:
            x, y, w, h = detection['bbox']
            cropped = frame[y:y+h, x:x+w]
            
            if cropped.size > 0:
                severity, severity_conf = self.severity_classifier.classify_severity(cropped)
                detection['severity'] = severity
                detection['severity_confidence'] = severity_conf
        
        # 3. Thermal Processing
        thermal_data = self.thermal_processor.simulate_thermal_data()
        fire_detected, fire_locations = self.thermal_processor.detect_fire(thermal_data)
        
        results['detections'] = detections
        results['thermal_fire'] = fire_detected
        results['fire_locations'] = fire_locations
        results['thermal_data'] = thermal_data
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def run_demo(self, duration: int = 30):
        """Run demo with webcam or simulated data"""
        print("Starting Integrated Detection System Demo...")
        print("Press 'q' to quit, 's' to show thermal data")
        
        # Try to open webcam, fallback to simulated data
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not available, using simulated data")
            cap = None
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    # Simulate frame
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw results
                display_frame = self.yolo_detector.draw_detections(frame, results['detections'])
                
                # Add system info
                fps = frame_count / (time.time() - start_time + 1e-6)
                info_text = f"FPS: {fps:.1f} | Detections: {len(results['detections'])}"
                if results['thermal_fire']:
                    info_text += f" | FIRE DETECTED: {len(results['fire_locations'])} locations"
                
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show severity info
                y_offset = 60
                for i, det in enumerate(results['detections']):
                    if 'severity' in det:
                        severity_text = f"Detection {i+1}: {det['severity']} ({det['severity_confidence']:.2f})"
                        cv2.putText(display_frame, severity_text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 25
                
                # Display
                cv2.imshow('Integrated Detection System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and results['thermal_fire']:
                    # Show thermal visualization
                    self.thermal_processor.visualize_thermal(
                        results['thermal_data'], 
                        results['fire_locations']
                    )
                
                frame_count += 1
                
                # Stop after duration
                if time.time() - start_time > duration:
                    break
                    
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\nDemo Statistics:")
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total runtime: {total_time:.2f}s")


def main():
    """Main function to run the system"""
    print("Initializing Integrated Detection System...")
    
    # Create system
    system = IntegratedDetectionSystem()
    
    # Run demo
    system.run_demo(duration=60)  # Run for 60 seconds
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
