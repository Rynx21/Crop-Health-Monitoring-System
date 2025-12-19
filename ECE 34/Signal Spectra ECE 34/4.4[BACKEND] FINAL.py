# app_refactor_with_controls_fixed.py
"""
Flask + Spectrogram-Driven Filter (optimized + MFCC + CSV log)
Save as: app_refactor_with_controls_fixed.py
"""

import os
import time
import threading
import csv
import json
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal
from scipy.signal import convolve2d as scipy_convolve2d
from scipy.io import wavfile
from pydub import AudioSegment
from flask import Flask, send_file, render_template_string, jsonify, request, redirect, url_for, make_response
import librosa
import librosa.display
import pandas as pd  # used by export_results_to_csv
import speech_recognition as sr

# ----------------------------
def butter_bandpass_filter(audio, sr, lowcut=100, highcut=8000, order=4):
    """Apply IIR Butterworth bandpass filter to audio."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    try:
        # Use filtfilt for zero-phase filtering (avoids phase distortion)
        filtered = signal.filtfilt(b, a, audio)
    except Exception:
        # Fallback to lfilter if filtfilt fails (e.g., very short signals)
        filtered = signal.lfilter(b, a, audio)
    return filtered

# Usage example (replace FIR filtering with IIR):
# filtered_audio = butter_bandpass_filter(audio, sr, lowcut=100, highcut=8000, order=4)
# You can swap this in place of the FIR filter for comparison.
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectrogram_app")

# ----------------------------
# Configuration / paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_FILE = os.path.join(SCRIPT_DIR, "audio.m4a")
IMAGE_FILE = os.path.join(SCRIPT_DIR, "image.jpeg")
AUDIO_COEFFS_FILE = os.path.join(SCRIPT_DIR, "audio_coeffs.csv")
IMAGE_COEFFS_FILE = os.path.join(SCRIPT_DIR, "image_coeffs.csv")
COEFFS_Q_SCALE = None  # Optional scaling for quantized coefficients

DEFAULT_AUDIO_SR = 48000
DEFAULT_AUDIO_DURATION = 6.0
STFT_NFFT = 1024
STFT_HOP = STFT_NFFT // 4
MIN_PEAK_DB = 12.0

AUDIO_OUT_DIR = os.path.join(SCRIPT_DIR, "static", "audio")
IMAGE_OUT_DIR = os.path.join(SCRIPT_DIR, "static", "images")
SAVED_DIR = os.path.join(SCRIPT_DIR, "saved_files")
LOG_CSV = os.path.join(SCRIPT_DIR, "processing_log.csv")

for d in (AUDIO_OUT_DIR, IMAGE_OUT_DIR, SAVED_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------
# Utility helpers
# ----------------------------
def save_copy(src_path, prefix="copy_"):
    if not os.path.exists(src_path):
        return
    dst = os.path.join(SAVED_DIR, prefix + os.path.basename(src_path))
    try:
        with open(src_path, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    except Exception as e:
        logger.warning("save_copy failed: %s", e)


def save_numpy(arr, name):
    out = os.path.join(SAVED_DIR, f"{name}.npy")
    try:
        np.save(out, arr)
    except Exception as e:
        logger.warning("save_numpy failed: %s", e)


def save_json(data, name):
    out = os.path.join(SAVED_DIR, f"{name}.json")
    try:
        with open(out, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.warning("save_json failed: %s", e)


def load_coeffs(path):
    if not os.path.exists(path):
        return None
    try:
        coeffs = np.loadtxt(path, delimiter=",", ndmin=1)
    except Exception as e:
        logger.warning("load_coeffs failed to read %s: %s", path, e)
        return None
    if COEFFS_Q_SCALE:
        coeffs = coeffs.astype(np.float64) / COEFFS_Q_SCALE
    norm = np.sum(np.abs(coeffs))
    if norm > 0:
        coeffs = coeffs / norm
    return coeffs


def load_audio_coeffs():
    return load_coeffs(AUDIO_COEFFS_FILE)


def load_image_coeffs():
    return load_coeffs(IMAGE_COEFFS_FILE)


def log_results(row, csvfile=LOG_CSV):
    r = dict(row)
    if "timestamp" in r:
        try:
            r["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["timestamp"]))
        except Exception:
            r["timestamp"] = str(r["timestamp"])
    new_file = not os.path.exists(csvfile)
    try:
        with open(csvfile, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=r.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(r)
    except Exception as e:
        logger.warning("log_results failed: %s", e)


def export_results_to_csv(results, out_path=None):
    """
    Convert the latest processing results into a CSV file.
    Handles nested dictionaries by flattening them.
    Advanced features (audio/image) are expanded into separate columns.
    """
    if out_path is None:
        out_path = os.path.join(SAVED_DIR, "results.csv")

    if not isinstance(results, dict) or len(results) == 0:
        return None

    flat = {}
    for key, val in results.items():
        if isinstance(val, dict):
            # Flatten advanced_features dictionary
            if key == "advanced_features":
                for feat_key, feat_val in val.items():
                    flat[feat_key] = feat_val
            else:
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, np.ndarray):
                        sub_v = sub_v.tolist()
                    if isinstance(sub_v, (list, tuple, dict)):
                        flat[f"{key}_{sub_k}"] = json.dumps(sub_v, default=str)
                    else:
                        flat[f"{key}_{sub_k}"] = sub_v
        else:
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if isinstance(val, (list, tuple, dict)):
                flat[key] = json.dumps(val, default=str)
            else:
                flat[key] = val

    try:
        df = pd.DataFrame([flat])
        df.to_csv(out_path, index=False)
    except Exception as e:
        logger.warning("export_results_to_csv failed: %s", e)
        return None

    save_copy(out_path)
    return out_path

def save_generated_file(src_path):
    """
    Save any newly generated file in SAVED_DIR without changing the filename.
    This keeps the output file in place while keeping a backup copy.
    """
    if not os.path.exists(src_path):
        return
    dst = os.path.join(SAVED_DIR, os.path.basename(src_path))
    try:
        with open(src_path, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    except Exception as e:
        logger.warning("save_generated_file failed: %s", e)

# ----------------------------
# Image histogram helpers
# ----------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_color_histogram(img, out_path, title="Color Histogram"):
    """
    Save color histogram of an image.
    img: BGR image (uint8) or single-channel grayscale (2D)
    out_path: full path where PNG will be saved
    """
    try:
        plt.figure(figsize=(6, 3))
        if img is None:
            plt.text(0.5, 0.5, "No image", ha='center', va='center')
            plt.axis('off')
        elif img.ndim == 2:
            # grayscale treated as single-channel plot
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, color='k')
            plt.xlim([0, 255])
            plt.xlabel('Intensity')
            plt.ylabel('Count')
            plt.title(title)
        else:
            # Convert BGR → RGB for accurate plotting
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            chans = cv2.split(img_rgb)
            colors = ('r', 'g', 'b')  # matches visual color of channels
            for chan, col in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                plt.plot(hist, color=col, label=f'{col.upper()} channel')
            plt.xlim([0, 255])
            plt.xlabel('Intensity')
            plt.ylabel('Count')
            plt.title(title)
            plt.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        logger.warning("save_color_histogram failed: %s", e)

def save_gray_histogram(img, out_path, title="Grayscale Histogram"):
    """
    Save grayscale histogram of an image
    """
    try:
        if img is None:
            plt.figure(figsize=(6,3))
            plt.text(0.5, 0.5, "No image", ha='center', va='center')
            plt.axis('off')
            plt.savefig(out_path)
            plt.close()
            return
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(6, 3))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.xlim([0, 255])
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        logger.warning("save_gray_histogram failed: %s", e)


def extract_advanced_image_features(raw_img, filtered_img):
    """
    Extract numerical image features for adaptive performance evaluation.
    Returns a dict with metrics for raw and filtered images.
    Metrics:
        - Gabor energy (simple sum of filtered responses)
        - Fourier spectrum energy
        - Edge density
    """
    features = {}

    # Helper: compute Gabor energy
    def gabor_energy(img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energies = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kern = cv2.getGaborKernel((21,21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            energies.append(np.sum(np.abs(fimg)))
        return float(np.mean(energies))

    # Helper: compute Fourier energy
    def fourier_energy(img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        return float(np.sum(magnitude_spectrum))

    # Helper: edge density
    def edge_density(img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size)

    # Raw image features
    features['raw_gabor_energy'] = gabor_energy(raw_img)
    features['raw_fourier_energy'] = fourier_energy(raw_img)
    features['raw_edge_density'] = edge_density(raw_img)

    # Filtered image features
    features['filtered_gabor_energy'] = gabor_energy(filtered_img)
    features['filtered_fourier_energy'] = fourier_energy(filtered_img)
    features['filtered_edge_density'] = edge_density(filtered_img)

    return features


def generate_and_save_histograms(raw_img, filtered_img, gray_img=None, edge_img=None):
    """
    Generate and save histograms to static/images:
      - hist_raw.png
      - hist_filtered.png
      - hist_gray.png
      - hist_edges.png
    Returns:
      dict with paths + computed numeric features for logging.
    """
    out_paths = {}
    try:
        # Use absolute paths based on SCRIPT_DIR
        hist_dir = os.path.join(SCRIPT_DIR, "static", "images")
        os.makedirs(hist_dir, exist_ok=True)
        hist_raw_path = os.path.join(hist_dir, "hist_raw.png")
        hist_filt_path = os.path.join(hist_dir, "hist_filtered.png")
        hist_gray_path = os.path.join(hist_dir, "hist_gray.png")
        hist_edges_path = os.path.join(hist_dir, "hist_edges.png")

        # Convert images to RGB for plotting. Support both color (BGR)
        # and single-channel grayscale inputs by converting gray -> RGB.
        if raw_img.ndim == 3:
            raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        else:
            raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)

        if filtered_img.ndim == 3:
            filt_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
        else:
            filt_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

        # Raw image histogram
        plt.figure(figsize=(6,4))
        for i, col in enumerate(['r','g','b']):
            hist, bins = np.histogram(raw_rgb[:,:,i].flatten(), bins=256, range=[0,256])
            plt.plot(bins[:-1], hist, color=col, marker='o', markersize=2, linestyle='-', label=f'{col.upper()} channel')
        plt.title("Raw Image Color Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(hist_raw_path)
        plt.close()

        # Filtered image histogram (black only)
        plt.figure(figsize=(6,4))
        # Compute histogram for the first channel or convert to grayscale
        gray_image = np.mean(filt_rgb, axis=2)  # average RGB to grayscale
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0,256])
        plt.plot(bins[:-1], hist, color='black', marker='o', markersize=2, linestyle='-')
        plt.title("Filtered Image Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(hist_filt_path)
        plt.close()


        # Grayscale histogram
        if gray_img is not None:
            plt.figure(figsize=(6,4))
            hist, bins = np.histogram(gray_img.flatten(), bins=256, range=[0,256])
            plt.plot(bins[:-1], hist, color='black', marker='o', markersize=2, linestyle='-')
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel value")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(hist_gray_path)
            plt.close()

        # Edge histogram
        if edge_img is not None:
            hist = np.zeros(256, dtype=int)
            unique, counts = np.unique(edge_img, return_counts=True)
            hist[unique] = counts
            x_vals = [0, 255]
            y_vals = [hist[0], hist[255]]
            plt.figure(figsize=(6,4))
            plt.bar(x_vals, y_vals, width=50, color='black', edgecolor='white')
            plt.xticks([0,255], ["Background (0)", "Edges (255)"])
            plt.ylabel("Pixel Count")
            plt.title("Edge Histogram")
            plt.tight_layout()
            plt.savefig(hist_edges_path)
            plt.close()

        out_paths = {
            "hist_raw": "/static/images/hist_raw.png",
            "hist_filtered": "/static/images/hist_filtered.png",
            "hist_gray": "/static/images/hist_gray.png" if gray_img is not None else None,
            "hist_edges": "/static/images/hist_edges.png" if edge_img is not None else None
        }

        # Compute numeric features
        features = extract_advanced_image_features(raw_img, filtered_img)

        # Combine for return
        return {**out_paths, **features}

    except Exception as e:
        logger.warning("generate_and_save_histograms failed: %s", e)
        return {}

# ----------------------------
# Audio & Image processing (with dynamic CSV) - FULL FIX
# ----------------------------
def load_audio_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".wav"):
        sr, data = wavfile.read(path)
        if data.dtype != np.float32:
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32)
        if data.ndim > 1:
            data = data[:, 0]
        return data, sr
    if path.lower().endswith((".m4a", ".mp3", ".aac")):
        seg = AudioSegment.from_file(path)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
            samples = samples.mean(axis=1)
        denom = float(2 ** (8 * seg.sample_width - 1))
        if denom == 0:
            denom = float(2 ** 15)
        samples /= denom
        return samples, seg.frame_rate
    raise ValueError("Unsupported audio format")


def record_audio(duration, sr=DEFAULT_AUDIO_SR):
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32", blocking=True)
        return audio.flatten()
    except Exception as e:
        logger.warning("record_audio failed: %s", e)
        return np.zeros(int(duration * sr), dtype=np.float32)


def noise_gate(audio, threshold_db=-40, window_ms=20, sr=DEFAULT_AUDIO_SR):
    """Apply noise gate to suppress very quiet signals (background noise)."""
    try:
        window_samples = int(sr * window_ms / 1000.0)
        rms_values = []
        for i in range(0, len(audio), window_samples):
            chunk = audio[i:i+window_samples]
            rms = np.sqrt(np.mean(chunk ** 2))
            rms_values.append(rms)
        
        # Convert threshold_db to linear RMS
        threshold_linear = 10 ** (threshold_db / 20.0)
        
        # Create gate mask
        gate_mask = np.zeros_like(audio)
        for i, rms in enumerate(rms_values):
            start = i * window_samples
            end = min(start + window_samples, len(audio))
            if rms > threshold_linear:
                gate_mask[start:end] = 1.0
            else:
                gate_mask[start:end] = 0.0  # Mute quiet regions
        
        # Smooth gate transitions to avoid clicks
        gate_smooth = np.convolve(gate_mask, np.hanning(window_samples // 2), mode='same')
        gate_smooth = np.clip(gate_smooth, 0, 1)
        return audio * gate_smooth
    except Exception as e:
        logger.warning("noise_gate failed: %s", e)
        return audio

def spectral_subtraction(mag_db, noise_profile_db=None, subtraction_factor=0.8):
    """Reduce noise by subtracting estimated noise spectrum (gaming mic style)."""
    try:
        # Estimate noise floor from quietest frequencies (bottom 10%)
        if noise_profile_db is None:
            noise_profile_db = np.percentile(mag_db, 10, axis=1, keepdims=True)
        
        # Subtract noise spectrum with scaling factor
        mag_db_reduced = mag_db - subtraction_factor * (noise_profile_db - np.min(mag_db, axis=1, keepdims=True))
        
        # Prevent over-subtraction (floor at -80 dB)
        mag_db_reduced = np.maximum(mag_db_reduced, -80)
        
        return mag_db_reduced
    except Exception as e:
        logger.warning("spectral_subtraction failed: %s", e)
        return mag_db

def spectral_brightening(mag_db, preserve_ratios=True):
    """Restore high-frequency content to prevent lo-fi degradation."""
    try:
        # Create frequency-dependent brightness boost (extremely subtle to avoid artifacts)
        num_freqs = mag_db.shape[0]
        freq_weights = np.linspace(1.0, 1.1, num_freqs)  # Minimal boost at high freqs (1.0-1.1x)
        
        # Apply very minimal boost curve to preserve natural character and avoid noise artifacts
        freq_weights = 1.0 + 0.15 * (freq_weights - 1.0)  # Ultra-minimal enhancement (0.15x factor)
        
        mag_db_bright = mag_db + 0.3 * np.log10(freq_weights[:, np.newaxis] + 1e-12)  # Reduced from 0.8 to 0.3
        
        return mag_db_bright
    except Exception as e:
        logger.warning("spectral_brightening failed: %s", e)
        return mag_db

def adaptive_compression(audio, threshold_db=-20, ratio=4.0):
    """Apply gentle multiband compression to balance frequency response without artifacts."""
    try:
        # Compute RMS in small windows for adaptive gain
        window_size = 2048
        rms_envelope = np.zeros_like(audio)
        
        for i in range(0, len(audio), window_size // 2):
            chunk = audio[i:i+window_size]
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
            
            # Convert to dB
            rms_db = 20 * np.log10(rms + 1e-12)
            
            # Soft knee compression
            if rms_db > threshold_db:
                gain_reduction_db = (rms_db - threshold_db) * (1.0 - 1.0/ratio)
                gain_linear = 10 ** (-gain_reduction_db / 20.0)
            else:
                gain_linear = 1.0
            
            # Apply smooth gain envelope
            end_idx = min(i + window_size // 2, len(audio))
            rms_envelope[i:end_idx] = gain_linear
        
        # Smooth envelope to prevent clicks
        rms_envelope = np.convolve(rms_envelope, np.hanning(128) / np.sum(np.hanning(128)), mode='same')
        
        return audio * rms_envelope
    except Exception as e:
        logger.warning("adaptive_compression failed: %s", e)
        return audio

def de_muffle_filter(audio, sr=DEFAULT_AUDIO_SR):
    """Remove low-frequency muffle caused by aggressive noise suppression."""
    try:
        # High-pass filter at 80 Hz with gentle roll-off (prevents boxiness)
        sos = signal.butter(4, 80, 'high', fs=sr, output='sos')
        audio_dehp = signal.sosfilt(sos, audio)
        
        # Gentle mid-boost around 2-4 kHz (clarity and presence)
        sos_mid = signal.butter(2, [2000, 4000], 'band', fs=sr, output='sos')
        mid_band = signal.sosfilt(sos_mid, audio_dehp)
        
        # Blend: 80% main + 20% mid boost for natural tone
        audio_enhanced = 0.8 * audio_dehp + 0.2 * mid_band
        
        return audio_enhanced
    except Exception as e:
        logger.warning("de_muffle_filter failed: %s", e)
        return audio

def clear_ai_cast_noise_cancellation(audio, sr=DEFAULT_AUDIO_SR, environment='quiet'):
    """Apply Clear AI Cast noise cancellation (SteelSeries GG inspired).
    Uses adaptive spectral gating with voice activity detection.
    
    Environment options:
    - 'quiet': Gentle noise reduction (office, home)
    - 'noisy': Aggressive noise reduction (coffee shop, classroom)
    - 'extreme': Maximum noise suppression (construction, traffic)
    """
    try:
        # Compute STFT for analysis
        f, t, Zxx = signal.stft(audio, sr, nperseg=512, noverlap=512-128, boundary=None)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Step 1: Estimate noise profile - more aggressive for noisy environments
        frame_energy = np.sum(mag**2, axis=0)  # Energy per frame
        
        if environment == 'extreme':
            noise_threshold = np.percentile(frame_energy, 35)  # Bottom 35% = noise
            noise_floor_percentile = 20
            wiener_floor = 0.03  # Very aggressive
            noise_suppression = 0.005  # Extreme suppression (0.5%)
            spectral_sub_factor = 0.20  # Heavy subtraction
        elif environment == 'noisy':
            noise_threshold = np.percentile(frame_energy, 30)  # Bottom 30% = noise (increased from 25%)
            noise_floor_percentile = 15
            wiener_floor = 0.05  # More aggressive (reduced from 0.08)
            noise_suppression = 0.015  # Heavy suppression (1.5%, increased from 3%)
            spectral_sub_factor = 0.12  # Heavier subtraction (increased from 0.08)
        else:  # 'quiet' (default)
            noise_threshold = np.percentile(frame_energy, 15)  # Bottom 15% = silence
            noise_floor_percentile = 10
            wiener_floor = 0.1  # Gentle
            noise_suppression = 0.05  # Moderate suppression
            spectral_sub_factor = 0.0  # No extra subtraction
        
        noise_frames = frame_energy < noise_threshold
        
        # Estimate noise spectrum from quiet frames
        if np.any(noise_frames):
            noise_profile = np.median(mag[:, noise_frames], axis=1, keepdims=True)
        else:
            noise_profile = np.percentile(mag, noise_floor_percentile, axis=1, keepdims=True)
        
        # Step 2: Voice Activity Detection (VAD) - detect speech vs noise
        speech_frames = ~noise_frames
        
        # Step 3: Adaptive noise suppression with smooth transitions
        # Use Wiener filter approach: preserve speech, suppress noise
        snr = mag / (noise_profile + 1e-12)
        gain = snr / (snr + 1.0)  # Wiener gain: smooth between 0 and 1
        gain = np.maximum(gain, wiener_floor)  # Floor adjusted for environment
        
        # Apply VAD-aware gain (stronger suppression during silence)
        gain_vad = np.where(
            speech_frames,
            gain,  # Preserve speech with computed gain
            gain * noise_suppression  # Environment-dependent suppression
        )
        
        mag_filtered = mag * gain_vad
        
        # Step 4: Spectral subtraction boost for extra noise reduction in noisy environments
        if spectral_sub_factor > 0:
            noise_over_subtraction = noise_profile * spectral_sub_factor
            mag_filtered = np.maximum(mag_filtered - noise_over_subtraction, mag_filtered * 0.005)
        
        # Step 5: Reconstruct with original phase
        Zxx_filtered = mag_filtered * np.exp(1j * phase)
        _, audio_filtered = signal.istft(Zxx_filtered, sr, nperseg=512, noverlap=512-128, input_onesided=True)
        
        # Trim to original length
        audio_filtered = audio_filtered[:len(audio)]
        
        return audio_filtered
    except Exception as e:
        logger.warning("clear_ai_cast_noise_cancellation failed: %s", e)
        return audio


def deep_voice_equalizer(audio, sr=DEFAULT_AUDIO_SR):
    """Apply Deep Voice preset EQ (SteelSeries GG inspired) for warm, rich, deep voice profile."""
    try:
        # Deep Voice EQ bands (inspired by SteelSeries GG):
        # 100 Hz: +8 dB (sub-bass warmth)
        # 200 Hz: +6 dB (bass presence)
        # 500 Hz: +3 dB (lower mid warmth)
        # 1000 Hz: +2 dB (fundamental presence)
        # 3000 Hz: +4 dB (clarity and presence)
        # 6000 Hz: +2 dB (upper midrange)
        # 10000 Hz: +1 dB (air and brightness)
        
        # Apply parametric EQ using cascaded second-order filters
        eq_audio = audio.copy()
        
        # Bass foundation (100 Hz boost, Q=1.0)
        try:
            sos_bass = signal.butter(2, 100, 'low', fs=sr, output='sos')
            bass_band = signal.sosfilt(sos_bass, eq_audio)
            eq_audio = eq_audio + 0.15 * bass_band  # Subtle +2dB equivalent blend
        except:
            pass
        
        # Low-mid warmth (200-500 Hz boost)
        try:
            sos_lowmid = signal.butter(2, [150, 600], 'band', fs=sr, output='sos')
            lowmid_band = signal.sosfilt(sos_lowmid, eq_audio)
            eq_audio = eq_audio + 0.12 * lowmid_band  # Subtle +3dB equivalent blend
        except:
            pass
        
        # Mid presence (1000 Hz gentle boost)
        try:
            sos_mid = signal.butter(2, [700, 1500], 'band', fs=sr, output='sos')
            mid_band = signal.sosfilt(sos_mid, eq_audio)
            eq_audio = eq_audio + 0.06 * mid_band  # Subtle +1dB equivalent blend
        except:
            pass
        
        # Upper mid clarity (2500-3500 Hz boost for presence)
        try:
            sos_upper = signal.butter(2, [2200, 3800], 'band', fs=sr, output='sos')
            upper_band = signal.sosfilt(sos_upper, eq_audio)
            eq_audio = eq_audio + 0.10 * upper_band  # Subtle +2dB equivalent blend
        except:
            pass
        
        # Gentle brightness (6-10 kHz subtle lift)
        try:
            sos_bright = signal.butter(2, [5500, 10500], 'band', fs=sr, output='sos')
            bright_band = signal.sosfilt(sos_bright, eq_audio)
            eq_audio = eq_audio + 0.04 * bright_band  # Subtle +1dB equivalent blend
        except:
            pass
        
        # Normalize to prevent clipping from additive EQ
        max_val = np.max(np.abs(eq_audio))
        if max_val > 1.0:
            eq_audio = eq_audio / max_val
        
        return eq_audio
    except Exception as e:
        logger.warning("deep_voice_equalizer failed: %s", e)
        return audio

def spectrogram_driven_filter(audio, sr=DEFAULT_AUDIO_SR, nfft=STFT_NFFT, hop=STFT_HOP, coeffs=None, noise_env='noisy'):
    """Apply audio filtering: noise gate → FIR → Clear AI Cast noise cancellation → soft limiting."""
    if audio is None or len(audio) == 0:
        return np.zeros(1, dtype=np.float32), (np.array([]), np.array([]), np.array([[]]))

    # STAGE 1: Noise gate (suppress very quiet background noise)
    try:
        audio = noise_gate(audio, threshold_db=-50, window_ms=20, sr=sr)
    except Exception as e:
        logger.warning("Noise gate failed: %s", e)

    # STAGE 2: Apply CSV-driven audio coefficients if provided (FIR bandpass)
    if coeffs is not None and getattr(coeffs, "size", 0) > 0:
        try:
            pad_len = len(coeffs)
            audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
            audio_conv = np.convolve(audio_padded, coeffs, mode='same')
            audio = audio_conv[pad_len:-pad_len]
        except Exception as e:
            logger.warning("Applying CSV audio coefficients failed: %s", e)

    # STAGE 3: Apply Clear AI Cast noise cancellation (Wiener filter + VAD) with environment setting
    try:
        audio = clear_ai_cast_noise_cancellation(audio, sr=sr, environment=noise_env)
    except Exception as e:
        logger.warning("Clear AI Cast noise cancellation failed: %s", e)
    
    # STAGE 4: Reconstruct final audio with soft limiting
    audio_rec = audio
    
    # Soft limiting to prevent clipping while preserving natural dynamics
    # Apply gain reduction smoothly for peaks above 0.9
    max_val = np.max(np.abs(audio_rec))
    if max_val > 0.9:
        # Soft knee compression: gradual reduction for peaks
        threshold = 0.8
        knee_width = 0.1
        target_max = 0.85
        
        # For values above threshold, apply soft compression
        audio_rec = np.where(
            np.abs(audio_rec) > threshold,
            np.sign(audio_rec) * (threshold + (np.abs(audio_rec) - threshold) * 0.3),
            audio_rec
        )
        
        # Final safety clip to prevent any distortion
        audio_rec = np.clip(audio_rec, -0.95, 0.95)
    
    return audio_rec, (np.array([]), np.array([]), np.array([[]]))


def compute_audio_features(y, sr):
    if y is None or len(y) < 2:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0}
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(((y[:-1] * y[1:]) < 0).sum() / (len(y) - 1))
    try:
        freqs, psd = signal.welch(y, sr, nperseg=min(1024, len(y)))
        centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    except Exception:
        centroid = 0.0
    return {"rms": rms, "zcr": zcr, "centroid": centroid}


def save_spectrogram_image(audio, sr, out_path):
    """Save a spectrogram image for the given audio."""
    try:
        f, t, Zxx = signal.stft(audio, sr, nperseg=STFT_NFFT, noverlap=STFT_NFFT - STFT_HOP, boundary=None)
        mag_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        plt.figure(figsize=(8, 3))
        plt.pcolormesh(t, f, mag_db, shading='auto')
        plt.title("Spectrogram")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        save_copy(out_path)
    except Exception as e:
        logger.warning("save_spectrogram_image failed: %s", e)


def save_mfcc_plot(audio, sr, out_path):
    """Save MFCC image for the given audio."""
    try:
        if audio is None or len(audio) == 0:
            mfccs = np.zeros((13, 1))
        else:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    except Exception as e:
        logger.warning("save_mfcc_plot failed to compute MFCCs: %s", e)
        mfccs = np.zeros((13, max(1, int(len(audio) / 512))))
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    save_copy(out_path)


def save_audio_graphs(audio_raw, audio_filtered, sr):
    """Generate all four audio images: raw/filtered spectrogram + raw/filtered MFCC."""
    try:
        save_spectrogram_image(audio_raw, sr, os.path.join(IMAGE_OUT_DIR, "spectrogram_raw.png"))
    except Exception as e:
        logger.warning("spectrogram_raw generation failed: %s", e)

    try:
        save_spectrogram_image(audio_filtered, sr, os.path.join(IMAGE_OUT_DIR, "spectrogram_filtered.png"))
    except Exception as e:
        logger.warning("spectrogram_filtered generation failed: %s", e)

    try:
        save_mfcc_plot(audio_raw, sr, os.path.join(IMAGE_OUT_DIR, "mfcc_raw.png"))
    except Exception as e:
        logger.warning("mfcc_raw generation failed: %s", e)

    try:
        save_mfcc_plot(audio_filtered, sr, os.path.join(IMAGE_OUT_DIR, "mfcc_filtered.png"))
    except Exception as e:
        logger.warning("mfcc_filtered generation failed: %s", e)


def write_wav_safe(path, audio, sr):
    """Write WAV safely with clipping and 16-bit conversion."""
    try:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        data_int16 = (audio_clipped * 32767.0).astype(np.int16)
        wavfile.write(path, sr, data_int16)
    except Exception as e:
        logger.warning("write_wav_safe failed: %s", e)


def load_image_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to load image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def capture_image(warmup_frames=5):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        try:
            cam.release()
        except:
            pass
        out = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(out, "NoCam", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return out
    frame = None
    for _ in range(warmup_frames):
        ok, frame = cam.read()
        if not ok:
            frame = None
            time.sleep(0.05)
    try:
        cam.release()
    except:
        pass
    if frame is None:
        out = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(out, "NoCam", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return out
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def image_filter_with_coeffs(img, coeffs=None):
    """Apply CSV-driven image filtering; fallback to Gaussian blur if coeffs invalid."""
    if coeffs is None or getattr(coeffs, "size", 0) < 1 or np.allclose(np.sum(np.abs(coeffs)), 0.0):
        return cv2.GaussianBlur(img, (5, 5), 0).astype(np.uint8)

    kern1d = coeffs.astype(np.float32) / max(np.sum(np.abs(coeffs)), 1e-12)
    kern2d = np.outer(kern1d, kern1d).astype(np.float32)
    filtered = np.zeros_like(img, dtype=np.float32)

    if img.ndim == 3:
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            out_ch = cv2.filter2D(ch.astype(np.float32), -1, kern2d)
            min_val, max_val = float(out_ch.min()), float(out_ch.max())
            if max_val > min_val + 1e-6:
                out_ch = 255.0 * (out_ch - min_val) / (max_val - min_val)
            else:
                out_ch = np.clip(out_ch, 0, 255)
            filtered[:, :, c] = out_ch
    else:
        out_ch = cv2.filter2D(img.astype(np.float32), -1, kern2d)
        min_val, max_val = float(out_ch.min()), float(out_ch.max())
        if max_val > min_val + 1e-6:
            filtered = 255.0 * (out_ch - min_val) / (max_val - min_val)
        else:
            filtered = np.clip(out_ch, 0, 255)

    return filtered.astype(np.uint8)

def auto_canny_edge_detection(img, sigma=0.33):
    # Robust Canny with fallback thresholds. Returns uint8 0/255 image.
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        v = np.median(gray)
    except Exception:
        v = 0
    lower = int(max(1, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    try:
        edges = cv2.Canny(gray, lower, upper)
    except Exception:
        edges = np.zeros_like(gray, dtype=np.uint8)

    # If no edges found, try common fixed thresholds
    if edges.sum() == 0:
        try:
            edges = cv2.Canny(gray, 50, 150)
        except Exception:
            edges = np.zeros_like(gray, dtype=np.uint8)

    # If still empty, try a more permissive low threshold
    if edges.sum() == 0:
        try:
            edges = cv2.Canny(gray, 10, 100)
        except Exception:
            edges = np.zeros_like(gray, dtype=np.uint8)

    # Final fallback: use Sobel magnitude to derive thresholds
    if edges.sum() == 0:
        try:
            sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobx ** 2 + soby ** 2)
            m = float(np.median(mag))
            l = int(max(1, m * 0.5))
            u = int(min(255, m * 3.0 + 50))
            edges2 = cv2.Canny(gray, max(1, l), max(2, u))
            if edges2.sum() > 0:
                edges = edges2
        except Exception:
            pass

    # Ensure binary 0/255 uint8 output
    try:
        edges_bin = (edges > 0).astype(np.uint8) * 255
    except Exception:
        edges_bin = np.zeros_like(gray, dtype=np.uint8)

    # Quick debug log
    try:
        logger.debug("auto_canny: median=%s lower=%s upper=%s edges=%s", v, lower, upper, int(edges_bin.sum()))
    except Exception:
        pass

    return edges_bin

def write_wav_safe(path, audio, sr):
    try:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        data_int16 = (audio_clipped * 32767.0).astype(np.int16)
        wavfile.write(path, sr, data_int16)
    except Exception as e:
        logger.warning("write_wav_safe failed: %s", e)
# ----------------------------
# Advanced Image Features
# ----------------------------
def compute_advanced_image_features(img):
    """
    Compute numerical image features for adaptive evaluation.
    Returns a dictionary of features.
    """
    if img is None:
        return {}

    import cv2
    import numpy as np

    # Ensure grayscale for texture analysis
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = {}

    # ---------- Gabor features ----------
    gabor_kernels = []
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.05, 0.1, 0.2]
    for theta in orientations:
        for freq in frequencies:
            ksize = 21
            sigma = 5.0
            lambd = 1.0 / freq
            gamma = 0.5
            phi = 0
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
            gabor_kernels.append(kern)

    gabor_responses = [cv2.filter2D(gray, cv2.CV_32F, k) for k in gabor_kernels]
    if gabor_responses:
        gabor_stack = np.stack(gabor_responses, axis=0)
        features['gabor_mean'] = float(gabor_stack.mean())
        features['gabor_std'] = float(gabor_stack.std())

    # ---------- Fourier spectrum ----------
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    features['fft_mean'] = float(np.mean(magnitude_spectrum))
    features['fft_std'] = float(np.std(magnitude_spectrum))

    # ---------- Edge / Laplacian statistics ----------
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0  # Normalize to 0-1
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    features['edge_mean'] = float(edges.mean())
    features['edge_std'] = float(edges.std())
    features['lap_mean'] = float(lap.mean())
    features['lap_std'] = float(lap.std())

    return features

# ----------------------------
# Processing Manager (Fixed)
# ----------------------------
class ProcessingManager:
    def __init__(self, audio_duration=DEFAULT_AUDIO_DURATION, sample_rate=DEFAULT_AUDIO_SR, use_files=False, filter_type='fir', noise_env='noisy'):
        self.audio_duration = float(audio_duration)
        self.sample_rate = int(sample_rate)
        self.use_files = bool(use_files)
        self.lock = threading.Lock()
        self.last_run = 0.0
        self.results = {}
        self.audio_coeffs = load_audio_coeffs()   # dynamic audio coefficients
        self.image_coeffs = load_image_coeffs()   # dynamic image coefficients
        self.last_capture_time = 0.0
        self.capture_cooldown = 3.0
        self.last_captured_img = None
        self.filter_type = filter_type
        self.noise_env = noise_env  # 'quiet', 'noisy', or 'extreme'

    def update_config(self, audio_duration=None, sample_rate=None, use_files=None, filter_type=None, noise_env=None):
        with self.lock:
            if audio_duration is not None:
                try:
                    self.audio_duration = float(audio_duration)
                except Exception:
                    pass
            if sample_rate is not None:
                try:
                    self.sample_rate = int(sample_rate)
                except Exception:
                    pass
            if use_files is not None:
                self.use_files = bool(use_files)
            if filter_type is not None:
                self.filter_type = filter_type
            if noise_env is not None:
                if noise_env in ('quiet', 'noisy', 'extreme'):
                    self.noise_env = noise_env
            logger.info(
                "Updated config: duration=%s sr=%s use_files=%s filter_type=%s noise_env=%s",
                self.audio_duration, self.sample_rate, self.use_files, self.filter_type, self.noise_env
            )

    def reload_coeffs(self):
        with self.lock:
            self.audio_coeffs = load_audio_coeffs()
            self.image_coeffs = load_image_coeffs()
            logger.info(
                "Reloaded audio_coeffs: %s, image_coeffs: %s",
                None if self.audio_coeffs is None else self.audio_coeffs.shape,
                None if self.image_coeffs is None else self.image_coeffs.shape
            )

    def run_pipeline(self):
        acquired = self.lock.acquire(timeout=5.0)
        if not acquired:
            logger.warning("Could not acquire lock for run_pipeline")
            return self.results

        tstart = time.time()
        try:
            # --- AUDIO PROCESSING ---
            if self.use_files:
                audio, sr_loaded = load_audio_file(AUDIO_FILE)
            else:
                audio = record_audio(self.audio_duration, self.sample_rate)
                sr_loaded = self.sample_rate

            if sr_loaded != self.sample_rate and len(audio) > 0:
                audio = signal.resample_poly(audio, self.sample_rate, sr_loaded)

            raw_audio_path = os.path.join(AUDIO_OUT_DIR, "recorded.wav")
            write_wav_safe(raw_audio_path, audio, self.sample_rate)
            save_copy(raw_audio_path)
            save_numpy(audio, "audio_raw")

            # Select filter type
            if getattr(self, 'filter_type', 'fir') == 'butter':
                filtered_audio = butter_bandpass_filter(audio, self.sample_rate, lowcut=100, highcut=8000, order=4)
            else:
                noise_env = getattr(self, 'noise_env', 'noisy')
                filtered_audio, _ = spectrogram_driven_filter(audio, sr=self.sample_rate, coeffs=self.audio_coeffs, noise_env=noise_env)
            filt_audio_path = os.path.join(AUDIO_OUT_DIR, "filtered.wav")
            write_wav_safe(filt_audio_path, filtered_audio, self.sample_rate)
            save_copy(filt_audio_path)
            save_numpy(filtered_audio, "audio_filtered")
            save_audio_graphs(audio, filtered_audio, self.sample_rate)

            # --- SPEECH-TO-TEXT (FILTERED AUDIO) ---
            recognizer = sr.Recognizer()
            speech_text = ""
            try:
                with sr.AudioFile(filt_audio_path) as source:  
                    audio_data = recognizer.record(source)
                    speech_text = recognizer.recognize_google(audio_data)
            except Exception as e:
                logger.warning(f"Speech-to-text failed: {e}")
                speech_text = ""

            # --- IMAGE PROCESSING ---
            current_time = time.time()
            capture_allowed = (current_time - self.last_capture_time) >= self.capture_cooldown

            if self.use_files:
                img = load_image_file(IMAGE_FILE)
            elif capture_allowed or self.last_captured_img is None:
                img = capture_image()
                self.last_capture_time = current_time
                self.last_captured_img = img.copy()
            else:
                img = self.last_captured_img.copy()

            raw_img_path = os.path.join(IMAGE_OUT_DIR, "captured_raw.jpg")
            cv2.imwrite(raw_img_path, img)
            save_copy(raw_img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray_path = os.path.join(IMAGE_OUT_DIR, "captured_gray.jpg")
            cv2.imwrite(gray_path, gray)

            # Filter the grayscale image (Raw -> Grayscale -> Filter)
            filtered_img = image_filter_with_coeffs(gray, self.image_coeffs)

            # Ensure filtered image is uint8 and save as 3-channel BGR for UI consistency
            if filtered_img.dtype != np.uint8:
                filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            filt_img_path = os.path.join(IMAGE_OUT_DIR, "filtered_image.jpg")
            if filtered_img.ndim == 2:
                save_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
            else:
                save_bgr = filtered_img
            cv2.imwrite(filt_img_path, save_bgr)
            save_copy(filt_img_path)
            save_numpy(filtered_img, "filtered_image")

            # Skip blur to preserve sharp edge details for face detection
            filt_for_edges = filtered_img

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                filt_for_edges = clahe.apply(filt_for_edges)
            except Exception:
                pass  # continue with original if CLAHE fails

            try:
                edges = auto_canny_edge_detection(filt_for_edges)
            except Exception:
                edges = np.zeros_like(gray)

            # Dilate edges before morphology to thicken weak features
            try:
                dilate_kernel = np.ones((3, 3), dtype=np.uint8)
                edges = cv2.dilate(edges, dilate_kernel, iterations=1)
            except Exception:
                pass

            # Dual morphology: close (dilate then erode) to fill gaps, then open to remove speckle
            try:
                kernel = np.ones((3, 3), dtype=np.uint8)
                edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                edges_morphed = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, kernel)
                if edges_morphed.sum() > 0:
                    edges = edges_morphed
            except Exception:
                pass

            # Apply unsharp masking to sharpen edge details
            try:
                blurred = cv2.GaussianBlur(edges, (3, 3), 0)
                edges = cv2.addWeighted(edges, 1.5, blurred, -0.5, 0)
                edges = np.clip(edges, 0, 255).astype(np.uint8)
            except Exception:
                pass

            # Ensure binary 0/255 uint8 for consistent display
            try:
                edges = (edges > 127).astype(np.uint8) * 255
            except Exception:
                edges = np.uint8(edges)
            edges_path = os.path.join(IMAGE_OUT_DIR, "edges.jpg")
            cv2.imwrite(edges_path, edges)
            save_numpy(edges, "edges_image")

            hist_paths = generate_and_save_histograms(
                raw_img=img,
                filtered_img=filtered_img,
                gray_img=gray,
                edge_img=edges
            )

            # --- FEATURE EXTRACTION ---
            feats_before = compute_audio_features(audio, self.sample_rate)
            feats_after = compute_audio_features(filtered_audio, self.sample_rate)

            audio_coeff_norm = float(np.sum(np.abs(self.audio_coeffs))) if self.audio_coeffs is not None else 0.0
            image_coeff_norm = float(np.sum(np.abs(self.image_coeffs))) if self.image_coeffs is not None else 0.0

            # --- ADVANCED IMAGE FEATURES ---
            def compute_advanced_image_features(img):
                if img is None:
                    return {}
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                features = {}

                # Gabor
                gabor_kernels, orientations, frequencies = [], [0, np.pi/4, np.pi/2, 3*np.pi/4], [0.05, 0.1, 0.2]
                for theta in orientations:
                    for freq in frequencies:
                        ksize, sigma, lambd, gamma, phi = 21, 5.0, 1.0/freq, 0.5, 0
                        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
                        gabor_kernels.append(kern)
                gabor_responses = [cv2.filter2D(gray, cv2.CV_32F, k) for k in gabor_kernels]
                if gabor_responses:
                    gabor_stack = np.stack(gabor_responses, axis=0)
                    features['gabor_mean'] = float(gabor_stack.mean())
                    features['gabor_std'] = float(gabor_stack.std())

                # FFT
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                features['fft_mean'] = float(np.mean(magnitude_spectrum))
                features['fft_std'] = float(np.std(magnitude_spectrum))

                # Edge / Laplacian
                edges_img = cv2.Canny(gray, 50, 150).astype(np.float32)/255.0
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                features['edge_mean'] = float(edges_img.mean())
                features['edge_std'] = float(edges_img.std())
                features['lap_mean'] = float(lap.mean())
                features['lap_std'] = float(lap.std())

                return features

            img_feats = compute_advanced_image_features(filtered_img)

            # --- FLATTEN EVERYTHING ---
            flat_features = {}

            # Audio before
            for k, v in feats_before.items():
                flat_features[f"advanced_features_audio_before_{k}"] = v
            # Audio after
            for k, v in feats_after.items():
                flat_features[f"advanced_features_audio_after_{k}"] = v
            # Coeff norms
            flat_features["advanced_features_audio_coeff_norm"] = audio_coeff_norm
            flat_features["advanced_features_image_coeff_norm"] = image_coeff_norm
            # Image features
            for k, v in img_feats.items():
                flat_features[f"advanced_features_image_{k}"] = v

            # Save JSON version
            save_json(flat_features, "features_summary_flat")

            log_results({
                "timestamp": time.time(),
                "audio_rms_before": feats_before.get("rms", 0.0),
                "audio_rms_after": feats_after.get("rms", 0.0)
            })

            # --- SAVE RESULTS ---
            elapsed = time.time() - tstart
            self.results = {
                "raw_audio": raw_audio_path,
                "filtered_audio": filt_audio_path,
                "spectrogram_raw": os.path.join(IMAGE_OUT_DIR, "spectrogram_raw.png"),
                "spectrogram_filtered": os.path.join(IMAGE_OUT_DIR, "spectrogram_filtered.png"),
                "mfcc_raw": os.path.join(IMAGE_OUT_DIR, "mfcc_raw.png"),
                "mfcc_filtered": os.path.join(IMAGE_OUT_DIR, "mfcc_filtered.png"),
                "captured_raw": raw_img_path,
                "captured_gray": gray_path,
                "filtered_image": filt_img_path,
                "edges": edges_path,
                "hist_raw": hist_paths.get("hist_raw"),
                "hist_filtered": hist_paths.get("hist_filtered"),
                "hist_gray": hist_paths.get("hist_gray"),
                "hist_edges": hist_paths.get("hist_edges"),
                "advanced_features": flat_features,
                "speech_text": speech_text,
                "elapsed_s": elapsed,
                "timestamp": time.time()
            }

            self.last_run = time.time()

            # --- Export flattened CSV ---
            export_results_to_csv(self.results)

            return self.results

        finally:
            if acquired:
                self.lock.release()

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
manager = ProcessingManager()

# Load index.html template (must exist)
try:
    with open(os.path.join(SCRIPT_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        INDEX_TEMPLATE = f.read()
except Exception:
    INDEX_TEMPLATE = "<html><body><h1>Index missing</h1></body></html>"

@app.route("/")
def index():
    last = manager.results.get("timestamp")
    last_run_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last)) if last else 'never'
    return render_template_string(INDEX_TEMPLATE,
                                  server_time=time.strftime('%H:%M:%S'),
                                  current_duration=int(manager.audio_duration),
                                  last_run=last_run_str,
                                  use_files=manager.use_files,
                                  filter_type=getattr(manager, 'filter_type', 'fir'),
                                  noise_env=getattr(manager, 'noise_env', 'noisy'),
                                  elapsed_s=(manager.results.get("elapsed_s") if manager.results else None))

@app.route("/apply_config", methods=["POST"])
def apply_config():
    try:
        dur = float(request.form.get('audio_duration', manager.audio_duration))
    except Exception:
        dur = manager.audio_duration
    use_files_raw = request.form.get('use_files')
    use_files_val = True if use_files_raw in ("on", "true", "1") else False
    filter_type = request.form.get('filter_type', 'fir')
    noise_env = request.form.get('noise_env', 'noisy')
    manager.update_config(audio_duration=dur, use_files=use_files_val, filter_type=filter_type, noise_env=noise_env)
    return redirect(url_for('index'))

@app.route("/export_csv")
def export_csv():
    path = export_results_to_csv(manager.results)
    if path is None:
        return jsonify({"error": "No results to export"}), 400
    return send_file(path, as_attachment=True)

@app.route("/features_summary")
def get_features_summary():
    json_path = os.path.join("static", "data", "features_summary.json")

    if not os.path.exists(json_path):
        return jsonify({"error": "features_summary.json not found"}), 404

    with open(json_path, "r") as f:
        data = json.load(f)

    return jsonify(data)

# ----------------------------
# Serve audio
# ----------------------------
def _no_cache_send(path, mimetype):
    resp = make_response(send_file(path, mimetype=mimetype))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route("/audio")
def audio():
    path = os.path.join(AUDIO_OUT_DIR, "filtered.wav")
    if not os.path.exists(path):
        return jsonify({"error": "no audio"}), 404
    return _no_cache_send(path, mimetype="audio/wav")

@app.route("/audio_raw")
def audio_raw():
    path = os.path.join(AUDIO_OUT_DIR, "recorded.wav")
    if not os.path.exists(path):
        return jsonify({"error": "no raw audio"}), 404
    return _no_cache_send(path, mimetype="audio/wav")

@app.route('/speech_text')
def get_speech_text():
    if hasattr(manager, 'results'):
        return jsonify({"speech_text": manager.results.get("speech_text", "")})
    return jsonify({"speech_text": ""})

# ----------------------------
# Serve images
# ----------------------------
@app.route("/mfcc")
def mfcc():
    path = os.path.join(IMAGE_OUT_DIR, "mfcc_filtered.png")
    if not os.path.exists(path):
        return jsonify({"error": "no filtered mfcc"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/mfcc_raw")
def mfcc_raw():
    path = os.path.join(IMAGE_OUT_DIR, "mfcc_raw.png")
    if not os.path.exists(path):
        return jsonify({"error": "no raw mfcc"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/spectrogram")
def spectrogram():
    path = os.path.join(IMAGE_OUT_DIR, "spectrogram_filtered.png")
    if not os.path.exists(path):
        return jsonify({"error": "no filtered spectrogram"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/spectrogram_raw")
def spectrogram_raw():
    path = os.path.join(IMAGE_OUT_DIR, "spectrogram_raw.png")
    if not os.path.exists(path):
        return jsonify({"error": "no raw spectrogram"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/captured_raw")
def captured_raw():
    path = os.path.join(IMAGE_OUT_DIR, "captured_raw.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "no raw image"}), 404
    return _no_cache_send(path, mimetype="image/jpeg")

@app.route("/captured_gray")
def captured_gray():
    path = os.path.join(IMAGE_OUT_DIR, "captured_gray.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "no gray image"}), 404
    return _no_cache_send(path, mimetype="image/jpeg")

@app.route("/filtered_image")
def filtered_image():
    path = os.path.join(IMAGE_OUT_DIR, "filtered_image.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "no filtered image"}), 404
    return _no_cache_send(path, mimetype="image/jpeg")

@app.route("/edges")
def edges():
    path = os.path.join(IMAGE_OUT_DIR, "edges.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "no edges image"}), 404
    return _no_cache_send(path, mimetype="image/jpeg")

# ----------------------------
# Histogram routes
# ----------------------------
@app.route("/hist_raw")
def hist_raw():
    path = os.path.join(IMAGE_OUT_DIR, "hist_raw.png")
    if not os.path.exists(path):
        return jsonify({"error": "no hist_raw"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/hist_filtered")
def hist_filtered():
    path = os.path.join(IMAGE_OUT_DIR, "hist_filtered.png")
    if not os.path.exists(path):
        return jsonify({"error": "no hist_filtered"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/hist_gray")
def hist_gray():
    path = os.path.join(IMAGE_OUT_DIR, "hist_gray.png")
    if not os.path.exists(path):
        return jsonify({"error": "no hist_gray"}), 404
    return _no_cache_send(path, mimetype="image/png")

@app.route("/hist_edges")
def hist_edges():
    path = os.path.join(IMAGE_OUT_DIR, "hist_edges.png")
    if not os.path.exists(path):
        return jsonify({"error": "no hist_edges"}), 404
    return _no_cache_send(path, mimetype="image/png")

# ----------------------------
# Force processing endpoint (synchronous)
# ----------------------------
@app.route("/force_process")
def force_process():
    results = manager.run_pipeline()
    ts = int(results.get("timestamp", time.time()))
    return jsonify({
        "status": "done",
        "elapsed_s": results.get("elapsed_s"),
        "images": {
            "captured_raw": url_for('captured_raw', _external=False) + f"?ts={ts}",
            "captured_gray": url_for('captured_gray', _external=False) + f"?ts={ts}",
            "filtered_image": url_for('filtered_image', _external=False) + f"?ts={ts}",
            "edges": url_for('edges', _external=False) + f"?ts={ts}",
            "spectrogram": url_for('spectrogram', _external=False) + f"?ts={ts}",
            "spectrogram_raw": url_for('spectrogram_raw', _external=False) + f"?ts={ts}",
            "mfcc": url_for('mfcc', _external=False) + f"?ts={ts}",
            "mfcc_raw": url_for('mfcc_raw', _external=False) + f"?ts={ts}",
            "hist_raw": url_for('hist_raw', _external=False) + f"?ts={ts}",
            "hist_filtered": url_for('hist_filtered', _external=False) + f"?ts={ts}",
            "hist_gray": url_for('hist_gray', _external=False) + f"?ts={ts}",
            "hist_edges": url_for('hist_edges', _external=False) + f"?ts={ts}"
        }
    })

# ----------------------------
# Convenience route to run processing and return to index
# ----------------------------
@app.route("/process")
def process():
    manager.run_pipeline()
    return redirect(url_for('index'))

# ----------------------------
# Status and CSV log
# ----------------------------
@app.route("/processing_log.csv")
def serve_processing_logs():
    csv_path = os.path.join(SCRIPT_DIR, "processing_log.csv")
    return send_file(csv_path, mimetype="text/csv")

@app.route("/status")
def status():
    return jsonify({
        "last_run": manager.last_run,
        "features_summary": manager.results.get("features") if manager.results else {},
        "speech_text": manager.results.get("speech_text", ""),
        "use_files": manager.use_files,
        "audio_duration": manager.audio_duration
    })

@app.route("/csv_log")
def csv_log():
    if not os.path.exists(LOG_CSV):
        return jsonify([])
    try:
        with open(LOG_CSV, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.warning("reading CSV failed: %s", e)
        rows = []
    return jsonify(rows)

if __name__ == "__main__":
    logger.info("Starting Flask app: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)