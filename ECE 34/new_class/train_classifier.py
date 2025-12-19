"""
YOLO Classifier Training Script
Trains YOLOv8 classification model for disease/quality classification

Usage:
    python train_classifier.py --data path/to/dataset --crop potato --epochs 50
"""

import argparse
import os
import torch
from ultralytics import YOLO

def train_classifier(data_path, crop_name, epochs=50, imgsz=224, batch_size=32, device='auto'):
    """
    Train YOLOv8 classifier model
    
    Args:
        data_path: Path to dataset folder (contains train/val/test subfolders)
        crop_name: Name of crop (potato, corn, rice, etc.)
        epochs: Number of training epochs
        imgsz: Image size (224 for classifier)
        batch_size: Batch size
        device: Device to use ('cpu' or GPU id like '0')
    """
    
    print(f"Starting classifier training for {crop_name}...")
    print(f"Dataset path: {data_path}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
    
    # Verify dataset structure
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset path {data_path} not found")
        exit(1)
    
    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"⚠️  Warning: {dir_path} not found (optional for test)")
    
    # Resolve device automatically unless explicitly provided
    if device == 'auto':
        resolved_device = '0' if torch.cuda.is_available() else 'cpu'
    else:
        resolved_device = str(device)
    print(f"Using device: {resolved_device} (CUDA available: {torch.cuda.is_available()})")

    # Load YOLOv8 small classification model (better accuracy than nano)
    model = YOLO('yolov8s-cls.pt')
    
    # Train model
    use_light_aug = os.environ.get('LIGHT_AUG', '0') == '1'
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=resolved_device,
        patience=15,  # Early stopping
        save=True,
        project=f'runs/classifier_{crop_name}',
        name='weights',
        verbose=True,
        augment=not use_light_aug,
        auto_augment='none' if use_light_aug else 'randaugment',
        erasing=0.0 if use_light_aug else 0.4,
        flipud=0.0 if use_light_aug else 0.5,
        fliplr=0.0 if use_light_aug else 0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=0,  # Disable multiprocessing to avoid spawn issues on Windows
        cache=False,  # Disable caching to avoid verification issues
        plots=False,  # Disable plotting threads to avoid Windows thread hangs
        deterministic=True,
        amp=False
    )
    
    # Save model with crop-specific name
    output_path = f'{crop_name}_classifier.pt'
    model.save(output_path)
    print(f"\n✅ Classifier training complete!")
    print(f"Model saved: {output_path}")
    
    # Print results
    print(f"\nTraining Results:")
    print(f"  Top-1 Accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
    print(f"  Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 'N/A')}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO classifier for crop disease/quality classification')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--crop', type=str, required=True, help='Crop name (potato, corn, rice, etc.)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', or GPU id like '0'")
    parser.add_argument('--light-aug', action='store_true', help='Use minimal augmentations for stability')
    
    args = parser.parse_args()
    
    # Verify data path exists
    if not os.path.exists(args.data):
        print(f"❌ Error: {args.data} not found")
        print(f"Expected dataset structure:")
        print(f"  {args.data}/")
        print(f"    ├── train/")
        print(f"    │   ├── class1/")
        print(f"    │   ├── class2/")
        print(f"    │   └── class3/")
        print(f"    ├── val/")
        print(f"    │   ├── class1/")
        print(f"    │   ├── class2/")
        print(f"    │   └── class3/")
        print(f"    └── test/  (optional)")
        exit(1)
    
    # Allow overriding augment style via CLI flag too
    if args.light_aug:
        os.environ['LIGHT_AUG'] = '1'
    else:
        os.environ['LIGHT_AUG'] = '0'

    train_classifier(
        data_path=args.data,
        crop_name=args.crop,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
