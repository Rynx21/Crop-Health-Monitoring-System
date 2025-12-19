# Crop Health and Monitoring System

A Flask-based web application for real-time crop disease detection and classification using YOLOv8. Supports multiple crops (Tomato, Potato, Chili, Rice) with both local webcam and ESP32-CAM integration, plus Arduino sensor monitoring.

## Features

- 🌱 Multi-crop disease detection and classification
- 📹 Real-time video streaming (local webcam or ESP32-CAM)
- 🔬 YOLOv8 object detection and classification
- 🌡️ Arduino sensor integration (temperature, humidity, soil moisture)
- 🌤️ Weather data integration (OpenWeatherMap)
- 📊 Firebase real-time database logging
- 🖼️ Configurable image enhancement
- 🥧 Raspberry Pi optimized

## Supported Crops(more crops will be supported in the future)

- **Tomato** - 10 disease classes
- **Potato** - 3 disease classes
- **Chili Pepper** - 2 disease classes
- **Rice** - 4 disease classes

## Quick Start

See [`ECE 34/new_class/README.md`](ECE%2034/new_class/README.md) for comprehensive setup, usage, and code documentation.

## Project Structure

```
ECE 34/
  new_class/           # Main application
    app.py             # Flask server
    crops_config.json  # Crop configuration
    README.md          # Full documentation
    *.py               # Training & evaluation scripts
    templates/         # HTML templates
    static/            # Static assets
```

## Installation

```powershell
# Clone repository
git clone <your-repo-url>
cd "ECE 34 (1)"

# Create virtual environment
python -m venv ECE\ 34/.venv
& "ECE 34\.venv\Scripts\Activate.ps1"

# Install dependencies
pip install ultralytics opencv-python flask pillow requests pyserial

# Optional (dataset tools)
# Only needed if you will download datasets via Kaggle
pip install kaggle

# Notes:
# - Ultralytics will install PyTorch automatically; GPU is used if available.
# - If using ESP32-CAM, no extra package is required (HTTP MJPEG stream).
# - For Raspberry Pi, these defaults are tuned for CPU-only performance.
```
**Downloads:**
1) Go to [Releases](https://github.com/Rynx21/Crop-Health-Monitoring-System/releases)
2) Download All Files
3) Put all files to `ECE 34/new_class/`

**Included models:**
- `detector.pt` (5.96 MB) - Main object detector
- `classifier.pt` (2.83 MB) - Fallback classifier
- `tomato_classifier.pt` (9.81 MB)
- `potato_classifier.pt` (9.78 MB)
- `chili_classifier.pt` (9.78 MB)
- `rice_classifier.pt` (9.80 MB)
- `yolov8s.pt` (21.54 MB) - YOLOv8 detection base
- `yolov8s-cls.pt` (12.26 MB) - YOLOv8 classification base

**Dataset Links:**
- **Tomato**: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Potato**: [Potato Disease Classification](https://www.kaggle.com/datasets/vipomonozon/potato-disease-classification)
- **Chili**: [PlantVillage Pepper Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Rice**: [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/tedmylo/ricerice-disease-image-dataset)

For training, organize datasets as:
```
crop_classifier_dataset/
  train/
    Class1/
    Class2/
  val/
    Class1/
    Class2/
```

## Usage

```powershell
# Run the server
& "ECE 34\.venv\Scripts\python.exe" "ECE 34\new_class\app.py"

# Train rice classifier (example)
& "ECE 34\.venv\Scripts\python.exe" "ECE 34\new_class\train_rice_enhanced.py"

# Evaluate model accuracy
& "ECE 34\.venv\Scripts\python.exe" "ECE 34\new_class\evaluate_model_accuracy.py"
```

## Configuration

Environment variables:
- `ENABLE_IMAGE_ENHANCEMENT`: Enable/disable image enhancement (default: true)
- `ENABLE_SERIAL_READER`: Enable Arduino sensor reading (default: true)
- `ESP32_URL`: ESP32-CAM MJPEG stream URL (optional)
### Install

```powershell
# Windows (PowerShell) — run from the repository root
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
Full documentation including architecture, API endpoints, and code walkthrough: [`ECE 34/new_class/README.md`](ECE%2034/new_class/README.md)
& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install ultralytics opencv-python flask pillow requests pyserial

MIT

## Acknowledgments

### Run

```powershell
$env:ENABLE_IMAGE_ENHANCEMENT = "true"
$env:ENABLE_SERIAL_READER = "true"
& ".\.venv\Scripts\python.exe" "ECE 34\new_class\app.py"
- PlantVillage Dataset
### Train Rice (optional)

```powershell
& ".\.venv\Scripts\python.exe" "ECE 34\new_class\train_rice_enhanced.py"
