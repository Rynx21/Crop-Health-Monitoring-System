"""
Model Accuracy Evaluation Script
Calculates and displays actual accuracy of trained YOLO classifier models
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import time

# Configuration
BASE_DIR = Path(__file__).parent
CROPS_CONFIG_FILE = BASE_DIR / 'crops_config.json'

# Load crop configuration
with open(CROPS_CONFIG_FILE, 'r', encoding='utf-8') as f:
    crops_config = json.load(f)


def find_validation_dataset(crop_name):
    """Find validation dataset for a crop"""
    # Search in multiple possible locations
    root = BASE_DIR.parent.parent  # ECE 34 (1) folder
    
    possible_paths = [
        # In archived_files
        BASE_DIR / 'archived_files' / 'unused_datasets' / f'{crop_name}_classifier_dataset' / 'val',
        # In root
        root / f'{crop_name}_classifier_dataset' / 'val',
        # In archived_datasets at root
        root / 'archived_datasets' / f'{crop_name}_classifier_dataset' / 'val',
        # In ECE 34 parent folder
        BASE_DIR.parent / f'{crop_name}_classifier_dataset' / 'val',
    ]
    
    # Also search recursively if not found
    for path in possible_paths:
        if path.exists() and path.is_dir():
            # Check if it has subdirectories (class folders)
            if any(path.iterdir()):
                return path
    
    # Last resort: search recursively in root
    try:
        for item in root.rglob(f'{crop_name}_classifier_dataset'):
            val_path = item / 'val'
            if val_path.exists() and val_path.is_dir() and any(val_path.iterdir()):
                return val_path
    except Exception:
        pass
    
    return None


def evaluate_classifier(model_path, val_dir):
    """Evaluate classifier model on validation dataset"""
    if not val_dir or not val_dir.exists():
        return None
    
    print(f"  Loading model: {model_path.name}")
    model = YOLO(str(model_path))
    
    # Get all image files from validation subdirectories
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images_by_class = defaultdict(list)
    
    # Collect all images organized by class
    for class_dir in val_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    images_by_class[class_name].append(img_file)
    
    if not images_by_class:
        print("  ⚠ No validation images found")
        return None
    
    # Run inference and collect results
    total_correct = 0
    total_images = 0
    class_stats = {}
    
    print(f"  Evaluating {sum(len(imgs) for imgs in images_by_class.values())} images...")
    
    for true_class, image_files in images_by_class.items():
        correct = 0
        predictions = defaultdict(int)
        
        for img_path in image_files:
            try:
                results = model.predict(str(img_path), verbose=False, imgsz=224)
                pred_id = int(results[0].probs.top1)
                pred_class = results[0].names[pred_id]
                confidence = float(results[0].probs.top1conf)
                
                predictions[pred_class] += 1
                
                if pred_class == true_class:
                    correct += 1
                    
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
                continue
        
        total_images += len(image_files)
        total_correct += correct
        
        class_accuracy = (correct / len(image_files) * 100) if len(image_files) > 0 else 0
        class_stats[true_class] = {
            'total': len(image_files),
            'correct': correct,
            'accuracy': class_accuracy,
            'predictions': dict(predictions)
        }
    
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_images': total_images,
        'total_correct': total_correct,
        'class_stats': class_stats
    }


def display_results(crop_name, results):
    """Display evaluation results in a formatted way"""
    if not results:
        print(f"  ⚠ No results available\n")
        return
    
    print(f"\n  {'='*60}")
    print(f"  OVERALL ACCURACY: {results['overall_accuracy']:.2f}%")
    print(f"  Total Images: {results['total_images']}")
    print(f"  Correct Predictions: {results['total_correct']}")
    print(f"  {'='*60}")
    
    print(f"\n  Per-Class Performance:")
    print(f"  {'-'*60}")
    
    for class_name, stats in sorted(results['class_stats'].items()):
        print(f"  📊 {class_name}")
        print(f"     Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        
        if len(stats['predictions']) > 1:
            print(f"     Predictions breakdown:")
            for pred_class, count in sorted(stats['predictions'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total'] * 100)
                marker = "✓" if pred_class == class_name else "✗"
                print(f"       {marker} {pred_class}: {count} ({percentage:.1f}%)")
        print()


def main():
    print("="*70)
    print("MODEL ACCURACY EVALUATION")
    print("="*70)
    print()
    
    results_summary = []
    
    for crop_id, crop_data in crops_config['crops'].items():
        if not crop_data.get('enabled', True):
            continue
        
        print(f"\n{'='*70}")
        print(f"🌱 Evaluating {crop_data['name']} ({crop_id})")
        print(f"{'='*70}")
        
        classifier_model = crop_data.get('classifier_model')
        if not classifier_model:
            print("  ⚠ No classifier model configured")
            continue
        
        model_path = BASE_DIR / classifier_model
        if not model_path.exists():
            print(f"  ⚠ Model file not found: {classifier_model}")
            continue
        
        # Find validation dataset
        val_dir = find_validation_dataset(crop_id)
        if not val_dir:
            print(f"  ⚠ Validation dataset not found for {crop_id}")
            print(f"     Expected: {crop_id}_classifier_dataset/val/")
            print(f"     Searched in:")
            print(f"       - archived_files/unused_datasets/")
            print(f"       - Root directory and subdirectories")
            print(f"     → Please ensure validation dataset exists with class subfolders")
            continue
        
        print(f"  📁 Validation dataset: {val_dir.relative_to(BASE_DIR.parent.parent)}")
        
        # Evaluate model
        start_time = time.time()
        results = evaluate_classifier(model_path, val_dir)
        elapsed = time.time() - start_time
        
        if results:
            display_results(crop_id, results)
            print(f"  ⏱ Evaluation time: {elapsed:.2f}s")
            
            results_summary.append({
                'crop': crop_data['name'],
                'accuracy': results['overall_accuracy'],
                'images': results['total_images']
            })
    
    # Display summary
    if results_summary:
        print(f"\n{'='*70}")
        print("SUMMARY OF ALL MODELS")
        print(f"{'='*70}\n")
        
        for item in sorted(results_summary, key=lambda x: x['accuracy'], reverse=True):
            print(f"  {item['crop']:20s} | Accuracy: {item['accuracy']:6.2f}% | Images: {item['images']:4d}")
        
        avg_accuracy = sum(x['accuracy'] for x in results_summary) / len(results_summary)
        print(f"\n  {'='*60}")
        print(f"  Average Accuracy Across All Models: {avg_accuracy:.2f}%")
        print(f"  {'='*60}\n")


if __name__ == '__main__':
    main()
