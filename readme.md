# Minimal Detection & Classification System

A Python demonstration system that simulates the core functionality of YOLO object detection, ResNet severity classification, and AMG8833 thermal fire detection for educational and prototyping purposes.

## 🚀 What's Implemented

### YOLO-Style Object Detection
- **Simulated Detection**: Generates realistic bounding boxes with confidence scores
- **Real-time Visualization**: Draws detection boxes on video feed
- **Confidence Filtering**: Configurable threshold-based detection filtering
- **Performance Monitoring**: Live FPS counter and timing metrics

### ResNet18 Severity Classification
- **PyTorch Implementation**: Uses pre-trained ResNet18 from torchvision
- **Image Preprocessing**: Proper normalization and resizing for inference
- **Multi-class Output**: Classifies incidents as Low/Medium/High severity
- **Confidence Scoring**: Returns classification confidence levels

### AMG8833 Thermal Simulation
- **8x8 Grid Simulation**: Mimics real AMG8833 thermal sensor output
- **Fire Detection Logic**: 150°C threshold-based fire detection
- **DBSCAN Clustering**: Groups fire pixels and filters noise
- **Thermal Visualization**: Matplotlib heatmap with fire location markers

### Integrated Demo System
- **Live Processing**: Real-time frame processing pipeline
- **Webcam Support**: Uses system camera with simulation fallback
- **Interactive Controls**: Keyboard commands for system control
- **Performance Display**: Shows FPS, detection counts, and processing time

## 📋 Requirements

```
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

## 🔧 Installation & Usage

1. **Install dependencies**
```bash
pip install opencv-python torch torchvision numpy scikit-learn matplotlib
```

2. **Run the demo**
```bash
python detection_system.py
```

3. **Interactive controls during demo**
- **'q'**: Quit the demo
- **'s'**: Show thermal heatmap when fire is detected
- **Ctrl+C**: Force stop

## 🎮 Demo Features

### What You'll See
- **Live video window** with simulated object detections
- **Bounding boxes** around detected incidents
- **Severity labels** (Low/Medium/High) with confidence scores
- **Fire alerts** when thermal simulation detects high temperatures
- **Real-time statistics** showing FPS and detection counts

### Sample Output
```
Starting Integrated Detection System Demo...
Press 'q' to quit, 's' to show thermal data

Demo Statistics:
Total frames processed: 847
Average FPS: 14.12
Total runtime: 60.00s
```

## 🏗️ Code Structure

### Core Classes

**`YOLODetector`**
- Simulates object detection with random bounding boxes
- Implements confidence thresholding
- Provides visualization utilities

**`ResNetSeverityClassifier`**
- Real PyTorch ResNet18 model (pre-trained)
- Custom final layer for 3-class severity classification
- Proper image preprocessing pipeline

**`ThermalProcessor`**
- Generates realistic 8x8 thermal data
- Implements fire detection at 150°C threshold
- Uses DBSCAN for clustering fire pixels

**`IntegratedDetectionSystem`**
- Orchestrates the complete processing pipeline
- Handles webcam input and visualization
- Manages performance monitoring

### Processing Pipeline

```
Input Frame → YOLO Detection → Crop Detections → ResNet Classification
     ↓                                                    ↓
Thermal Data → Fire Detection → Clustering → Results Display
```

## ⚠️ Important Notes

### This is a Demo System
- **YOLO detections are simulated** - not from a real trained model
- **Thermal data is synthetic** - not from actual AMG8833 sensor
- **ResNet classifier** uses pre-trained weights but isn't trained on incident data
- **Performance metrics** are for demonstration purposes

### Real Implementation Would Need
- Actual YOLO model trained on your 5K+ labeled frames
- ResNet fine-tuned on your specific incident severity dataset
- Real AMG8833 sensor integration
- Production-grade error handling and logging

## 🔧 Customization

### Modify Detection Behavior
```python
# Adjust detection simulation parameters
detector = YOLODetector(confidence_threshold=0.7)  # Higher threshold
```

### Change Fire Detection Sensitivity
```python
# Modify thermal processing
thermal = ThermalProcessor(fire_threshold=120.0)  # Lower temperature
```

### Adjust Demo Duration
```python
# Run for different time periods
system.run_demo(duration=30)  # 30 seconds instead of 60
```

## 📊 Performance Notes

- **FPS**: Typically 10-20 FPS depending on system capabilities
- **CPU Usage**: Moderate - ResNet inference is the heaviest operation
- **Memory**: ~2-4GB RAM usage for model loading
- **GPU**: Will use CUDA if available, falls back to CPU

## 🚨 Limitations

### Current Constraints
- Simulated detections don't reflect real-world accuracy
- No actual training data integration
- Limited to demonstration purposes
- No persistent storage or logging
- Basic error handling only

### Not Suitable For
- Production deployment
- Actual fire safety systems
- Real-time critical applications
- Performance benchmarking of actual models

## 🛠️ Extending the Demo

### To Make It Production-Ready
1. Replace simulated YOLO with actual trained model
2. Fine-tune ResNet on your severity classification dataset
3. Integrate real AMG8833 sensor hardware
4. Add proper logging and error handling
5. Implement data persistence and alerts
6. Add configuration management

### Hardware Integration Example
```python
# For real AMG8833 integration (not implemented)
import board
import adafruit_amg88xx

i2c = board.I2C()
amg = adafruit_amg88xx.AMG88XX(i2c)
# thermal_data = amg.pixels  # Real sensor data
```

## 📁 File Structure

```
main.py    # Complete demo implementation
README.md             # This documentation
```

## 🎯 Educational Value

This demo shows:
- How to structure a multi-component computer vision system
- PyTorch model integration patterns
- Real-time video processing techniques
- Thermal data simulation and processing
- System performance monitoring
- Interactive demo development

Perfect for understanding the architecture before implementing with real trained models and hardware sensors.

---

**Note: This is a demonstration system showing the integration pattern of your described components, not the actual production system with trained models.**
