# Crop Health and Monitoring System (new_class)

A concise guide to set up, run, train, and evaluate the crop disease detection/classification app in this folder.

## Overview

- Flask server streams video, detects and classifies leaf diseases, logs results, and optionally reads Arduino sensors.
- Supports local webcam or ESP32-CAM MJPEG stream.
- Uses Ultralytics YOLOv8 for detection and classification.

## Requirements

- Python 3.10+ recommended
- Packages: ultralytics, opencv-python, flask, pillow, requests, pyserial (optional for Arduino)

Install into your existing virtual environment:

```powershell
& "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\.venv\Scripts\python.exe" -m pip install ultralytics opencv-python flask pillow requests pyserial
# Optional: Kaggle (only if you'll use import_datasets.py to download datasets)
& "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\.venv\Scripts\python.exe" -m pip install kaggle
```

## Configuration

- Crop models and class names: `crops_config.json`
- Default crop: `default_crop` key in `crops_config.json`
- Image enhancement:
  - Enable/disable via `ENABLE_IMAGE_ENHANCEMENT` environment variable (default: true)
  - Coefficients file: `image_coeffs.csv` (fallback path in Signal Spectra project if not found here)
- Arduino serial reader: toggle via `ENABLE_SERIAL_READER` (default: true)
- ESP32-CAM MJPEG stream: set `ESP32_URL` (e.g., `http://<ip>:81/stream`)
- Weather and Firebase keys/URLs are defined in `app.py` (consider moving to environment variables for production).

## Run the Server

```powershell
# Optional toggles
$env:ENABLE_IMAGE_ENHANCEMENT = "true"
$env:ENABLE_SERIAL_READER = "true"
# $env:ESP32_URL = "http://<esp32-ip>:81/stream"  # uncomment to use ESP32-CAM

& "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\.venv\Scripts\python.exe" "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\new_class\app.py"
```

- Access the web UI via your browser (default Flask host/port; update `app.py` if you want a specific host/port).
- If neither webcam nor ESP32 is available, a placeholder image is served.

## Datasets & Class Names

Your classifier datasets must follow this layout (example: rice):

```
ECE 34/new_class/rice_classifier_dataset/
  train/
    BrownSpot/
    Healthy/
    Hispa/
    LeafBlast/
  val/
    BrownSpot/
    Healthy/
    Hispa/
    LeafBlast/
```

Class folder names must match keys in `leaf_classes` for the crop in `crops_config.json`.

## Train the Rice Classifier

- Ensure dataset exists at `ECE 34/new_class/rice_classifier_dataset` as shown above.
- Start training:

```powershell
& "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\.venv\Scripts\python.exe" "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\new_class\train_rice_enhanced.py"
```

- Model: `yolov8s-cls.pt` backbone; outputs saved under `runs/classifier_rice/weights`.
- Auto-selects GPU if available, otherwise CPU.

## Evaluate Model Accuracy

- Evaluates all enabled crops against discovered `val/` datasets.

```powershell
& "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\.venv\Scripts\python.exe" "C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\new_class\evaluate_model_accuracy.py"
```

- Prints per-class and overall accuracy summaries.

## ESP32-CAM Integration

- Set `ESP32_URL` to your stream URL (e.g., `http://<ip>:81/stream`).
- The app will automatically use ESP32 frames when configured.

## Arduino Sensor Integration

- With `ENABLE_SERIAL_READER=true`, the app scans common COM ports and reads CSV-like lines such as `temperature:25.6,humidity:60,soil:450`.
- Latest values are kept in memory and can be exposed via routes/UI (customizable in `app.py`).

## Image Enhancement

- Enabled by default; uses `image_coeffs.csv` when present to build a kernel.
- Falls back to a pre-defined sharpen kernel otherwise.
- Disable if you need raw frames: `$env:ENABLE_IMAGE_ENHANCEMENT = "false"`.

## Logs & Outputs

- Detections: `detection_log.csv` (timestamp, crop, detection type, class, confidence, status)
- Firebase: detection results also PATCHed to your configured Realtime DB URL
- Training runs: `runs/` (created by Ultralytics)

## Troubleshooting

- Camera unavailable: Check USB, try removing `cv2.CAP_DSHOW` (Windows) or use ESP32.
- Slow performance (Pi): Frame skipping and smaller image sizes are already applied; keep enhancement only if needed.
- Arduino upload conflicts: Temporarily set `$env:ENABLE_SERIAL_READER = "false"`.
- Class mismatches: Ensure folder names match `leaf_classes` keys exactly.

## Extending to New Crops

1. Place your detector/classifier `.pt` files in this folder.
2. Add a new crop entry in `crops_config.json` with model filenames and `leaf_classes` mapping.
3. Ensure dataset class folder names match the mapping.
4. Train a classifier and update the config to use its output `.pt`.

---

For questions or improvements, see `app.py`, `evaluate_model_accuracy.py`, `train_rice_enhanced.py`, and `import_datasets.py` for reference implementations.

## How the Code Works (Comprehensive Guide)

### Architecture

- **Server:** Flask app in `app.py` orchestrates camera/ESP32 input, inference, logging, sensors, and REST endpoints.
- **Models:** Ultralytics YOLOv8 detector (`detector.pt`) and per-crop classifiers (e.g., `tomato_classifier.pt`, `rice_classifier.pt`).
- **Config:** `crops_config.json` defines enabled crops, model filenames, and class name mappings.
- **Pipelines:** Frames → optional enhancement → detection (boxes) → classification (per box or full-frame) → logging + Firebase.

### Key Modules

- [`app.py`](app.py): Main server
  - Platform tuning (Raspberry Pi defaults for speed) — lines [17-24](app.py#L17-L24)
  - Image enhancement via `image_coeffs.csv` (or fallback sharpen) — [`load_image_coeffs()`](app.py#L39), [`apply_image_enhancement()`](app.py#L57)
  - Camera init (Windows DirectShow, Linux/Pi V4L2) — [`init_camera()`](app.py#L137)
  - ESP32 MJPEG stream support — [`_esp32_frame_iter()`](app.py#L341)
  - Arduino serial thread for sensors — [`start_serial_reader()`](app.py#L173)
  - Weather routes using OpenWeatherMap — [`get_weather()`](app.py#L241), [`classify_icon()`](app.py#L292)
  - Detection/classification frame generator (`/video_feed`) — [`generate_frames()`](app.py#L365)
- `crops_config.json`: Per-crop JSON config
  - `detector_model`, `classifier_model`
  - `leaf_classes` mapping from model label → human readable
  - `enabled` flag and `default_crop`
- [`evaluate_model_accuracy.py`](evaluate_model_accuracy.py): Accuracy evaluator
  - Finds `val/` datasets across common locations — [`find_validation_dataset()`](evaluate_model_accuracy.py#L22)
  - Runs classifier on each image — [`evaluate_classifier()`](evaluate_model_accuracy.py#L57)
  - Computes per-class and overall accuracy — [`display_results()`](evaluate_model_accuracy.py#L129)
- [`train_rice_enhanced.py`](train_rice_enhanced.py): Training script
  - Expects dataset at `rice_classifier_dataset/train` and `val`
  - Trains YOLOv8s-classification with tuned params — [`train_rice_classifier()`](train_rice_enhanced.py#L14)
  - Saves to `runs/classifier_rice/weights`
- `import_datasets.py`: Dataset utilities
  - Kaggle setup/download helpers
  - Organizers to lay out detector/classifier datasets

### Configuration & Models

- At startup, the app loads [`crops_config.json`](crops_config.json) and sets `current_crop` — see [app.py#L92-94](app.py#L92-L94).
- [`load_crop_models(crop_name)`](app.py#L113) loads YOLO models from filenames defined in config.
- If a configured file is missing, it falls back to `detector.pt` / `classifier.pt` in the same folder — [app.py#L123-126](app.py#L123-L126).
- Class names returned by the model are mapped to friendly labels using `leaf_classes` from config.

### Camera & ESP32 Input

- **Local webcam:** Initialized with platform-friendly backends and resized to configured resolution — [`init_camera()`](app.py#L137-L167).
- **ESP32-CAM:** If `ESP32_URL` is set, the app reads MJPEG chunks and decodes frames — [`_esp32_frame_iter()`](app.py#L341-L362).
- If no source is available, a placeholder image is served so the UI stays responsive — see [app.py#L377-L379](app.py#L377-L379).

### Image Enhancement

- [`image_coeffs.csv`](image_coeffs.csv) is loaded at startup — [`load_image_coeffs()`](app.py#L39-L52); if present, it constructs a 2D kernel from coefficients.
- If coefficients are missing or invalid, a pre-defined sharpen kernel is used — see [`_SHARPEN_KERNEL`](app.py#L54-L56).
- Processing happens in [`apply_image_enhancement()`](app.py#L57-L76); toggle on/off via `ENABLE_IMAGE_ENHANCEMENT` — [app.py#L28](app.py#L28).

### Arduino Sensors

- The serial reader scans common ports, attempts to connect at `9600` baud, and reads CSV-like lines — [`start_serial_reader()`](app.py#L173-L217).
- Parsed keys (e.g., `temperature`, `humidity`, `soil`) update an in-memory `sensor_state` with the latest values and timestamp — [app.py#L202-L209](app.py#L202-L209).
- Toggle via `ENABLE_SERIAL_READER` — [app.py#L29](app.py#L29); thread starts at [app.py#L219-L223](app.py#L219-L223).

### Frame Processing & Inference

- The [`/video_feed`](app.py#L304) endpoint streams multipart JPEG frames generated from [`generate_frames()`](app.py#L365-L510).
- Per loop:
  - Acquire frame (camera or ESP32) — [app.py#L393-L405](app.py#L393-L405)
  - Apply enhancement if enabled — [app.py#L408](app.py#L408)
  - Skip frames (Pi-friendly defaults) to reduce compute load — [app.py#L411-L413](app.py#L411-L413)
  - Run detection (optional; caches last boxes/labels) — [app.py#L418-L431](app.py#L418-L431)
  - Run classification per box or full-frame — [app.py#L433-L498](app.py#L433-L498)
  - Encode JPEG and yield to the client — [app.py#L501-L510](app.py#L501-L510)

### Weather & Icons

- [`/weather`](app.py#L241) route retrieves current and forecast data from OpenWeatherMap — [app.py#L241-L289](app.py#L241-L289).
- [`classify_icon()`](app.py#L292) maps description text to basic icons (rainy/cloudy/sunny).

### Logging & Firebase Integration

- [`log_detection()`](app.py#L307) appends rows to [`detection_log.csv`](detection_log.csv) with timestamp, crop, detection type, class, confidence, and status.
- [`send_detection_to_firebase()`](app.py#L321) formats a JSON payload and PATCHes your Firebase Realtime DB endpoint — [app.py#L322-L339](app.py#L322-L339).
- Confidence is also mirrored as "accuracy" for display consistency — [app.py#L329-L330](app.py#L329-L330).

### Accuracy Evaluation

- `evaluate_model_accuracy.py` scans likely dataset locations for `val/` sets with class folders.
- It runs the corresponding crop classifier against each image and aggregates per-class stats.
- Outputs a summary of overall accuracy across all enabled crops.

### Training Workflow (Rice Example)

- Prepare dataset at `rice_classifier_dataset/train` and `val` with the exact class folders: `BrownSpot`, `Healthy`, `Hispa`, `LeafBlast`.
- Run `train_rice_enhanced.py` to train with tuned hyperparameters and augmentations.
- Check outputs in `runs/classifier_rice/weights` and update `crops_config.json` if you swap model filenames.
- Re-run the evaluator to confirm improved accuracy.

### Common Pitfalls

- **Class name mismatch:** Folder names must exactly match keys in `leaf_classes`.
- **Camera backend issues:** On Windows, DirectShow is preferred; if startup fails, try default OpenCV backend.
- **Performance on Pi:** Lower `imgsz`, frame skipping, and JPEG quality are tuned; keep enhancement only if helpful.
- **Kaggle downloads:** Some datasets require accepting terms; use `import_datasets.py` helpers and verify download success.

### Extending to New Crops

- Add your `.pt` files and update `crops_config.json` with model filenames and `leaf_classes`.
- Ensure datasets use matching class folder names.
- Train a classifier for the new crop and reference its weights in the config.