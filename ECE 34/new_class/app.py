from flask import Flask, request, jsonify, render_template, Response
import csv
import json
import os
import platform
import threading
import time
from datetime import datetime
from typing import Iterator

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

IS_PI = 'arm' in platform.machine().lower() or 'aarch' in platform.machine().lower()
# Raspberry Pi-friendly defaults (CPU-only)
DEVICE = 'cpu'
HALF = False
DETECT_IMGSZ = 416 if IS_PI else 640
CLASS_IMGSZ = 160 if IS_PI else 224
SKIP_FRAMES_DEFAULT = 10 if IS_PI else 4
CAM_RES = (320, 240) if IS_PI else (640, 480)
JPEG_QUALITY = 75 if IS_PI else 85

app = Flask(__name__)

# Image enhancement settings
ENABLE_IMAGE_ENHANCEMENT = os.environ.get('ENABLE_IMAGE_ENHANCEMENT', 'true').lower() == 'true'
# Optional: allow disabling Arduino serial reader (useful during Arduino uploads)
ENABLE_SERIAL_READER = os.environ.get('ENABLE_SERIAL_READER', 'true').lower() == 'true'
# Prefer local file next to app.py, fallback to Signal Spectra path
IMAGE_COEFFS_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_coeffs.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Signal Spectra ECE 34', 'image_coeffs.csv')
]

def load_image_coeffs():
    """Load sharpening coefficients from CSV file"""
    try:
        for path in IMAGE_COEFFS_CANDIDATES:
            if os.path.exists(path):
                coeffs = np.loadtxt(path, delimiter=',', dtype=np.float32)
                if coeffs.size > 0:
                    print(f"✓ Loaded image enhancement coefficients from {path}: {coeffs.shape}")
                    return coeffs
    except Exception as e:
        print(f"Could not load image coefficients: {e}")
    return None

# Pre-allocate sharpening kernel once for efficiency
_SHARPEN_KERNEL = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]], dtype=np.float32)

_SHARPEN_KERNEL_STRONG = np.array([[-1, -1, -1],
                                    [-1, 10, -1],
                                    [-1, -1, -1]], dtype=np.float32)

def preprocess_frame(frame):
    """
    Lightweight preprocessing applied to full frame before detection.
    RPi4: Basic contrast stretch (~1-2ms)
    Desktop: CLAHE for better lighting adaptation (~30-50ms)
    """
    if not ENABLE_IMAGE_ENHANCEMENT:
        return frame
    
    if IS_PI:
        # RPi4: Lightweight contrast stretch
        return cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    else:
        # Desktop: Can afford CLAHE for variable lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def apply_image_enhancement(img, coeffs=None):
    """
    Apply enhanced ROI processing with HSV boost and adaptive sharpening.
    Lightweight enough for RPi4 on small ROIs (~10-15ms per crop).
    """
    if not ENABLE_IMAGE_ENHANCEMENT:
        return img
    
    # HSV saturation boost for better disease/health discrimination (~2-3ms)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Enhance saturation (helps distinguish disease symptoms)
    s = cv2.multiply(s, 1.2)
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Gamma correction only on desktop (RPi4 skips to save time)
    if not IS_PI:
        v = np.power(v / 255.0, 0.9) * 255
        v = v.astype(np.uint8)
    
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Adaptive sharpening based on image blur detection (~3-5ms)
    variance = cv2.Laplacian(enhanced, cv2.CV_64F).var()
    if variance < 100:  # Blurry image needs stronger sharpening
        kernel = _SHARPEN_KERNEL_STRONG
    else:
        kernel = _SHARPEN_KERNEL
    
    # Apply sharpening
    if coeffs is None or getattr(coeffs, "size", 0) < 1 or np.allclose(np.sum(np.abs(coeffs)), 0.0):
        return cv2.filter2D(enhanced, -1, kernel).astype(np.uint8)
    
    # Use CSV coefficients to build 2D kernel
    kern1d = coeffs.astype(np.float32) / max(np.sum(np.abs(coeffs)), 1e-12)
    kern2d = np.outer(kern1d, kern1d).astype(np.float32)
    out_ch = cv2.filter2D(enhanced.astype(np.float32), -1, kern2d)
    
    min_val, max_val = float(out_ch.min()), float(out_ch.max())
    if max_val > min_val + 1e-6:
        return (255.0 * (out_ch - min_val) / (max_val - min_val)).astype(np.uint8)
    return np.clip(out_ch, 0, 255).astype(np.uint8)

# Load enhancement coefficients at startup
image_enhancement_coeffs = load_image_coeffs()

# OpenWeatherMap API
API_KEY = 'd41485bcf1b4d828c3a14689a4c9f7c6'
CURRENT_URL = 'https://api.openweathermap.org/data/2.5/weather'
FORECAST_URL = 'https://api.openweathermap.org/data/2.5/forecast'

# Firebase Realtime Database URL
FIREBASE_URL = 'https://smart-agriculture-a88c9-default-rtdb.asia-southeast1.firebasedatabase.app/sensors.json'

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load crop configuration
CONFIG_FILE = os.path.join(BASE_DIR, 'crops_config.json')
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    crops_config = json.load(f)

# Optional ESP32 camera URL (set env var ESP32_URL, e.g., http://<ip>:81/stream)
ESP32_URL = os.environ.get('ESP32_URL')

# Detection log file
LOG_FILE = os.path.join(BASE_DIR, 'detection_log.csv')
log_lock = threading.Lock()

# Initialize log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Crop Type', 'Detection Type', 'Detection', 'Confidence', 'Accuracy', 'Status'])

# Current crop selection and models
current_crop = crops_config['default_crop']
models_lock = threading.Lock()

def load_crop_models(crop_name):
    """Load YOLO models for the specified crop"""
    crop_data = crops_config['crops'].get(crop_name)
    if not crop_data or not crop_data.get('enabled', True):
        return None, None
    
    detector_path = os.path.join(BASE_DIR, crop_data['detector_model'])
    classifier_path = os.path.join(BASE_DIR, crop_data['classifier_model'])
    
    # Check if models exist, fallback to default if not
    if not os.path.exists(detector_path):
        detector_path = os.path.join(BASE_DIR, 'detector.pt')
    if not os.path.exists(classifier_path):
        classifier_path = os.path.join(BASE_DIR, 'classifier.pt')
    
    detector = YOLO(detector_path)
    classifier = YOLO(classifier_path)
    return detector, classifier

# Load initial models
detector, classifier = load_crop_models(current_crop)

# Optimize camera settings (Pi-friendly)
# Initialize camera if available
def init_camera():
    try:
        # If ESP32 stream is configured, skip local camera init
        if ESP32_URL:
            print(f"ESP32 camera configured at {ESP32_URL} - skipping local webcam init")
            return None
        cap = None
        # Prefer DirectShow on Windows to avoid long startup delays
        if os.name == 'nt':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(0)
        else:
            # Prefer V4L2 on Linux/Raspberry Pi
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(0)

        if cap and cap.isOpened():
            w, h = CAM_RES
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera initialized: {w}x{h} @ ~30fps")
            return cap
        else:
            print("Camera not available - using placeholder stream")
            return None
    except Exception as e:
        print("Camera initialization error:", e)
        return None

camera = init_camera()

# Optional Arduino serial sensor support
sensor_state = {"temperature": None, "humidity": None, "soil": None, "port": None, "last_update": None}
def start_serial_reader():
    try:
        import serial
        import serial.tools.list_ports
        
        print("Scanning for Arduino...")
        arduino_port = None
        for port in serial.tools.list_ports.comports():
            if 'arduino' in port.description.lower() or 'CH340' in port.description or 'USB Serial' in port.description:
                arduino_port = port.device
                print(f"✓ Found Arduino on {arduino_port}: {port.description}")
                break
        
        if not arduino_port:
            ports = ['/dev/ttyACM0', '/dev/ttyUSB0', 'COM3', 'COM4', 'COM5']
            for p in ports:
                try:
                    test_ser = serial.Serial(p, 9600, timeout=1)
                    test_ser.close()
                    arduino_port = p
                    print(f"✓ Found device on {arduino_port}")
                    break
                except Exception:
                    continue
        
        if not arduino_port:
            print("✗ No Arduino detected")
            return
        
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        sensor_state["port"] = arduino_port
        time.sleep(2)
        
        print(f"Listening for sensor data on {arduino_port}...")
        while True:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    parts = [s.strip() for s in line.split(',')]
                    for part in parts:
                        if ':' in part:
                            k, v = part.split(':', 1)
                            try:
                                sensor_state[k.lower()] = float(v)
                            except ValueError:
                                sensor_state[k.lower()] = v
                    sensor_state["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                time.sleep(0.05)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.5)
    except Exception as e:
        print(f"Serial reader error: {e}")

if ENABLE_SERIAL_READER:
    serial_thread = threading.Thread(target=start_serial_reader, daemon=True)
    serial_thread.start()
else:
    print("Serial reader disabled via ENABLE_SERIAL_READER=false")

# Thread lock for camera access
camera_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    city = data.get('city', 'Oton,Iloilo,PH')

    loc_params = {'appid': API_KEY, 'units': 'metric'}
    if lat and lon:
        loc_params['lat'] = lat
        loc_params['lon'] = lon
    else:
        loc_params['q'] = city

    current_resp = requests.get(CURRENT_URL, params=loc_params)
    if current_resp.status_code != 200:
        return jsonify({'error': 'Failed to fetch current weather'}), 404

    current = current_resp.json()
    today = {
        'temp': current['main']['temp'],
        'min': current['main']['temp_min'],
        'main': current['weather'][0]['main'],
        'description': current['weather'][0]['description'],
        'icon': classify_icon(current['weather'][0]['description'])
    }

    forecast_resp = requests.get(FORECAST_URL, params=loc_params)
    if forecast_resp.status_code != 200:
        return jsonify({'error': 'Failed to fetch forecast', 'forecast': []}), 404

    forecast_data = forecast_resp.json().get('list', [])
    forecast_list = []
    seen_dates = set()

    for item in forecast_data:
        if len(forecast_list) >= 4:
            break
        dt = datetime.fromtimestamp(item['dt'])
        date_key = dt.strftime('%Y-%m-%d')
        
        if date_key not in seen_dates:
            desc = item['weather'][0]['description']
            forecast_list.append({
                'timestamp': item['dt'] * 1000,
                'temp': item['main']['temp'],
                'description': desc,
                'icon': classify_icon(desc)
            })
            seen_dates.add(date_key)
    return jsonify({'today': today, 'forecast': forecast_list})

def classify_icon(desc):
    desc = desc.lower()
    if 'rain' in desc or 'drizzle' in desc or 'storm' in desc:
        return 'rainy'
    elif 'cloud' in desc or 'overcast' in desc:
        return 'cloudy'
    elif 'clear' in desc or 'sun' in desc:
        return 'sunny'
    else:
        return 'cloudy'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def log_detection(crop_type, detection_type, class_name, confidence=None):
    try:
        with log_lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status = "Detected" if class_name != "no_detected" else "No Detection"
            conf_str = f"{confidence:.2f}" if confidence else "N/A"
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, crop_type, detection_type, class_name, conf_str, conf_str, status])
    except Exception as e:
        print("Error logging detection:", e)

# Send detection result to Firebase
def send_detection_to_firebase(crop_type, detection_type, class_name, confidence=None):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            "detection": {
                "crop_type": crop_type,
                "detection_type": detection_type,
                "result": class_name,
                "confidence": f"{confidence:.2f}" if confidence else "N/A",
                "accuracy": f"{confidence:.2f}" if confidence else "N/A",
                "timestamp": timestamp
            }
        }
        response = requests.patch(FIREBASE_URL, json=data)
        print("Firebase response:", response.status_code, response.text)
        log_detection(crop_type, detection_type, class_name, confidence)
    except Exception as e:
        print("Error sending detection to Firebase:", e)

# Detection with multi-crop support and leaf + fruit detection
def _esp32_frame_iter(url: str) -> Iterator:
    """Yield frames from an ESP32-CAM MJPEG stream."""
    while True:
        try:
            resp = requests.get(url, stream=True, timeout=5)
            bytes_buf = b''
            for chunk in resp.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                bytes_buf += chunk
                a = bytes_buf.find(b'\xff\xd8')
                b = bytes_buf.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg = bytes_buf[a:b+2]
                    bytes_buf = bytes_buf[b+2:]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    yield frame
        except Exception as e:
            print("ESP32 stream error:", e)
            time.sleep(1.0)
            yield None


def generate_frames():
    global detector, classifier, current_crop
    
    esp32_iter = None
    if not camera and ESP32_URL:
        esp32_iter = _esp32_frame_iter(ESP32_URL)
    elif camera is None and not ESP32_URL:
        # Return a placeholder frame if neither camera nor ESP32 is available
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open(os.path.join(BASE_DIR, 'static', 'placeholder.jpg'), 'rb').read() + b'\r\n'
        return
    
    last_sent = ""
    frame_count = 0
    skip_frames = SKIP_FRAMES_DEFAULT  # Process every Nth frame for detection
    last_boxes = []  # Cache last detection boxes
    last_labels = []  # Cache last labels
    last_firebase_send = time.time()  # Rate limit Firebase updates
    
    # Pre-allocate JPEG encode parameters
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    
    while True:
        if camera is not None:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                time.sleep(0.01)
                continue
        else:
            # ESP32 mode
            if esp32_iter is None:
                esp32_iter = _esp32_frame_iter(ESP32_URL)
            frame = next(esp32_iter, None)
            if frame is None:
                time.sleep(0.05)
                continue

        frame_count += 1
        detection_sent = False

        # Only run YOLO detection every 5th frame
        if frame_count % skip_frames == 0:
            with models_lock:
                active_crop = current_crop
                crop_data = crops_config['crops'][active_crop]
                leaf_classes = crop_data.get('leaf_classes', {})
                fruit_classes = crop_data.get('fruit_classes', {})
            
            # Apply preprocessing to full frame BEFORE detection (improves detection accuracy)
            frame_preprocessed = preprocess_frame(frame)
            
            # Convert to RGB once
            frame_rgb = cv2.cvtColor(frame_preprocessed, cv2.COLOR_BGR2RGB)
            
            # Run detector on reduced resolution (CPU-only)
            with models_lock:
                results = detector.predict(frame_rgb, conf=0.5, verbose=False, imgsz=DETECT_IMGSZ, half=HALF, device=DEVICE)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                # Clear cached boxes for new detection
                last_boxes = []
                last_labels = []

                # Limit to top 3 detections for speed
                top_indices = np.argsort(scores)[-3:][::-1]

                for idx in top_indices:
                    if scores[idx] < 0.5:
                        continue

                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    detected_item = frame[y1:y2, x1:x2]
                    if detected_item.size == 0:
                        continue
                    
                    # Apply enhanced ROI processing (HSV boost + adaptive sharpening)
                    detected_item = apply_image_enhancement(detected_item, image_enhancement_coeffs)
                    
                    # Direct RGB conversion and resize with better quality interpolation
                    item_rgb = cv2.cvtColor(detected_item, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(item_rgb, (CLASS_IMGSZ, CLASS_IMGSZ), interpolation=cv2.INTER_CUBIC)
                    pil_item = Image.fromarray(resized)

                    with models_lock:
                        result = classifier.predict(pil_item, verbose=False, imgsz=CLASS_IMGSZ, half=HALF, device=DEVICE)
                    
                    class_id = int(result[0].probs.top1)
                    class_name = result[0].names[class_id]
                    conf = result[0].probs.top1conf.item()

                    if conf < 0.7:
                        continue

                    # Determine if it's a leaf or fruit detection
                    detection_type = "leaf" if class_name in leaf_classes else "fruit"
                    display_name = leaf_classes.get(class_name, fruit_classes.get(class_name, class_name))
                    
                    label = f'{display_name} ({conf:.2f})'
                    color = (0, 255, 0) if detection_type == "leaf" else (255, 165, 0)  # Green for leaf, orange for fruit

                    # Cache boxes and labels
                    last_boxes.append((x1, y1, x2, y2, color))
                    last_labels.append(label)

                    # Rate limit Firebase updates (max once per 2 seconds)
                    current_time = time.time()
                    if class_name != last_sent:
                        if current_time - last_firebase_send >= 2.0:
                            send_detection_to_firebase(active_crop, detection_type, class_name, conf)
                            last_sent = class_name
                            last_firebase_send = current_time
                            detection_sent = True

                # Send "no_detected" if nothing valid was detected (rate limited)
                if not detection_sent and last_sent != "no_detected":
                    current_time = time.time()
                    if current_time - last_firebase_send >= 2.0:
                        send_detection_to_firebase(active_crop, "none", "no_detected", None)
                        last_sent = "no_detected"
                        last_firebase_send = current_time
        
        # Draw cached boxes and labels on every frame for smooth display
        for i, box_data in enumerate(last_boxes):
            if len(box_data) == 5:
                x1, y1, x2, y2, color = box_data
            else:
                x1, y1, x2, y2 = box_data[:4]
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if i < len(last_labels):
                cv2.putText(frame, last_labels[i], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Optimized JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/arduino_status')
def arduino_status():
    """Check Arduino connection status"""
    if sensor_state.get("port"):
        return jsonify({
            "connected": True,
            "port": sensor_state.get("port"),
            "last_update": sensor_state.get("last_update"),
            "data": {
                "temperature": sensor_state.get("temperature"),
                "humidity": sensor_state.get("humidity"),
                "soil": sensor_state.get("soil")
            }
        })
    return jsonify({"connected": False})

@app.route('/sensor_data')
def sensor_data_route():
    try:
        # Prefer local Arduino serial if available
        if sensor_state.get("soil") is not None or sensor_state.get("temperature") is not None:
            return jsonify({
                "temperature": sensor_state.get("temperature"),
                "humidity": sensor_state.get("humidity"),
                "soil": sensor_state.get("soil"),
                "source": "arduino"
            })
        # Fallback to Firebase
        response = requests.get(FIREBASE_URL, timeout=3)
        if response.status_code == 200:
            data = response.json() or {}
            return jsonify({
                "temperature": data.get("temperature"),
                "humidity": data.get("humidity"),
                "soil": data.get("soil_moisture") or data.get("soil"),
                "source": "firebase"
            })
        else:
            return jsonify({"error": "Failed to fetch from Firebase"}), 500
    except Exception as e:
        print("Firebase error:", e)
        return jsonify({"error": "Exception occurred"}), 500

@app.route('/detection_logs')
def get_detection_logs():
    """Get all detection logs"""
    try:
        def _read_logs(limit=None):
            entries = []
            expected_keys = ['Timestamp', 'Crop Type', 'Detection Type', 'Detection', 'Confidence', 'Accuracy', 'Status']
            with log_lock:
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, 'r', newline='') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        # Skip header row if present
                        start_idx = 1 if rows and rows[0] and rows[0][0] == 'Timestamp' else 0
                        data_rows = rows[start_idx:]
                        if limit is not None and limit > 0:
                            data_rows = data_rows[-limit:]
                        for r in data_rows:
                            # Normalize row length to 6
                            if not r:
                                continue
                            # Backward compatibility: old rows had 6 columns (no Accuracy)
                            if len(r) == 6:
                                # r: [ts, crop, type, det, conf, status] -> insert conf as accuracy before status
                                r = [r[0], r[1], r[2], r[3], r[4], r[4], r[5]]
                            if len(r) < 7:
                                r = (r + [''] * 7)[:7]
                            elif len(r) > 7:
                                r = r[:7]
                            entry = dict(zip(expected_keys, r))
                            entries.append(entry)
            return entries

        logs = _read_logs(limit=None)
        return jsonify({"logs": logs, "total": len(logs)})
    except Exception as e:
        print("Error reading logs:", e)
        return jsonify({"error": "Failed to read logs"}), 500

@app.route('/detection_logs/latest/<int:count>')
def get_latest_logs(count):
    """Get latest N detection logs"""
    try:
        def _read_logs(limit):
            entries = []
            expected_keys = ['Timestamp', 'Crop Type', 'Detection Type', 'Detection', 'Confidence', 'Accuracy', 'Status']
            with log_lock:
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, 'r', newline='') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        # Skip header row if present
                        start_idx = 1 if rows and rows[0] and rows[0][0] == 'Timestamp' else 0
                        data_rows = rows[start_idx:]
                        if limit is not None and limit > 0:
                            data_rows = data_rows[-limit:]
                        for r in data_rows:
                            if not r:
                                continue
                            if len(r) == 6:
                                r = [r[0], r[1], r[2], r[3], r[4], r[4], r[5]]
                            if len(r) < 7:
                                r = (r + [''] * 7)[:7]
                            elif len(r) > 7:
                                r = r[:7]
                            entry = dict(zip(expected_keys, r))
                            entries.append(entry)
            return entries

        logs = _read_logs(count)
        return jsonify({"logs": logs, "count": len(logs)})
    except Exception as e:
        print("Error reading logs:", e)
        return jsonify({"error": "Failed to read logs"}), 500

@app.route('/detection_logs/clear', methods=['POST'])
def clear_logs():
    """Clear all detection logs"""
    try:
        with log_lock:
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Crop Type', 'Detection Type', 'Detection', 'Confidence', 'Accuracy', 'Status'])
        return jsonify({"message": "Logs cleared successfully"})
    except Exception as e:
        print("Error clearing logs:", e)
        return jsonify({"error": "Failed to clear logs"}), 500

@app.route('/crops')
def get_crops():
    """Get all available crops"""
    available_crops = []
    for crop_id, crop_data in crops_config['crops'].items():
        if crop_data.get('enabled', True):
            available_crops.append({
                'id': crop_id,
                'name': crop_data['name'],
                'icon': crop_data['icon']
            })
    return jsonify({
        'crops': available_crops,
        'current': current_crop
    })

@app.route('/set_crop', methods=['POST'])
def set_crop():
    """Set the active crop type"""
    global current_crop, detector, classifier
    
    data = request.get_json()
    crop_id = data.get('crop_id')
    
    if not crop_id or crop_id not in crops_config['crops']:
        return jsonify({"error": "Invalid crop type"}), 400
    
    crop_data = crops_config['crops'][crop_id]
    if not crop_data.get('enabled', True):
        return jsonify({"error": "Crop type not enabled"}), 400
    
    try:
        # Load new models
        new_detector, new_classifier = load_crop_models(crop_id)
        
        if new_detector is None or new_classifier is None:
            return jsonify({"error": "Failed to load models for this crop"}), 500
        
        # Update models with thread safety
        with models_lock:
            current_crop = crop_id
            detector = new_detector
            classifier = new_classifier
        
        return jsonify({
            "message": f"Successfully switched to {crop_data['name']}",
            "crop": crop_id,
            "crop_name": crop_data['name']
        })
    except Exception as e:
        print(f"Error switching crop: {e}")
        return jsonify({"error": "Failed to switch crop"}), 500

@app.route('/current_crop')
def get_current_crop():
    """Get currently active crop"""
    with models_lock:
        active_crop = current_crop
    
    crop_data = crops_config['crops'][active_crop]
    return jsonify({
        'crop_id': active_crop,
        'crop_name': crop_data['name'],
        'icon': crop_data['icon'],
        'leaf_classes': crop_data.get('leaf_classes', {}),
        'fruit_classes': crop_data.get('fruit_classes', {})
    })

@app.route('/classify/<crop>', methods=['POST'])
def classify_image(crop):
    """Classify a leaf/fruit image for the specified crop"""
    try:
        # Check if crop is enabled
        if crop not in crops_config['crops']:
            return jsonify({'error': f'Crop {crop} not found'}), 404
        
        crop_data = crops_config['crops'][crop]
        if not crop_data.get('enabled', True):
            return jsonify({'error': f'Crop {crop} is disabled'}), 400
        
        # Get image from request
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Load the classifier model for this crop
        classifier_path = os.path.join(BASE_DIR, crop_data['classifier_model'])
        if not os.path.exists(classifier_path):
            return jsonify({'error': f'Classifier model not found for {crop}'}), 500
        
        # Read and process image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Apply image enhancement
        img = apply_image_enhancement(img, image_enhancement_coeffs)
        
        # Prepare image for classification
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (CLASS_IMGSZ, CLASS_IMGSZ), interpolation=cv2.INTER_CUBIC)
        img_pil = Image.fromarray(img_resized)
        
        # Run inference
        crop_classifier = YOLO(classifier_path)
        result = crop_classifier.predict(img_pil, verbose=False, imgsz=CLASS_IMGSZ, half=HALF, device=DEVICE)
        
        # Extract prediction
        class_id = int(result[0].probs.top1)
        class_name = result[0].names[class_id]
        confidence = float(result[0].probs.top1conf.item())
        
        # Map class name to display name
        leaf_classes = crop_data.get('leaf_classes', {})
        fruit_classes = crop_data.get('fruit_classes', {})
        display_name = leaf_classes.get(class_name, fruit_classes.get(class_name, class_name))
        
        # Determine detection type
        detection_type = 'leaf' if class_name in leaf_classes else 'fruit'
        
        # Log detection
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status = 'success' if confidence >= 0.7 else 'low_confidence'
        with log_lock:
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, crop, detection_type, display_name, f'{confidence:.4f}', f'{confidence:.4f}', status])
        
        return jsonify({
            'crop': crop,
            'class_id': class_id,
            'class_name': class_name,
            'display_name': display_name,
            'detection_type': detection_type,
            'confidence': confidence,
            'confidence_pct': f'{confidence * 100:.2f}%',
            'accuracy': confidence,
            'accuracy_pct': f'{confidence * 100:.2f}%',
            'timestamp': timestamp
        })
    
    except Exception as e:
        print(f"Error classifying image for {crop}: {e}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
