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

## Supported Crops

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
```

## Model Files & Datasets

**Model files (`.pt`) are not included in this repository due to size.**

Download trained models:
- Place detector and classifier `.pt` files in `ECE 34/new_class/`
- Update `crops_config.json` with your model filenames

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

## Documentation

Full documentation including architecture, API endpoints, and code walkthrough: [`ECE 34/new_class/README.md`](ECE%2034/new_class/README.md)

## License

MIT

## Acknowledgments

- Ultralytics YOLOv8
- PlantVillage Dataset
- OpenWeatherMap API
