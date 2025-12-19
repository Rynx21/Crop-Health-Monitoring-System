#!/usr/bin/env python3
"""
Enhanced rice classifier training script with better error handling
"""
import sys
import os
import time

sys.path.insert(0, r"C:\Users\FSOS\Desktop\ECE 34 (1)\ECE 34\new_class")

from ultralytics import YOLO
import torch

def train_rice_classifier():
    print("=" * 60)
    print("RICE CLASSIFIER TRAINING (Target: >95% accuracy)")
    print("=" * 60)
    
    # Dataset configuration (use local project folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "rice_classifier_dataset")
    epochs = 25  # Increase from 20 to 25 for better convergence
    batch_size = 32  # Reduce from 64 to avoid memory issues
    imgsz = 224
    # Auto-select device: GPU if available, else CPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Data: {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {imgsz}")
    print(f"  Device: {('CUDA:'+str(device)) if device != 'cpu' else 'CPU'}")
    
    # Verify dataset
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"\n✗ Dataset not found!")
        return False
    
    print(f"\n✓ Dataset verified")
    
    try:
        # Load pretrained model
        model = YOLO("yolov8s-cls.pt")
        print("✓ Model loaded")
        
        # Train with optimized parameters
        print(f"\nStarting training...")
        print(f"Target: Top-1 Accuracy > 95%")
        
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=10,  # Early stopping patience
            save=True,
            cache=False,
            workers=0,
            amp=False,  # Disable AMP for GTX 1650
            optimizer='AdamW',
            lr0=0.001,  # Slightly lower learning rate
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            augment=True,
            # Data augmentation
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            # Output
            project="runs/classifier_rice",
            name="weights",
            exist_ok=True,
            verbose=True
        )
        
        # Check final accuracy
        if hasattr(results, 'results_dict'):
            accuracy = results.results_dict.get('metrics/accuracy_top1', 0)
            print(f"\n✓ Training completed!")
            print(f"Final Accuracy: {accuracy * 100:.2f}%")
            
            if accuracy > 0.95:
                print(f"✅ TARGET ACHIEVED! Accuracy > 95%")
            else:
                print(f"⚠️ Target not reached. Accuracy: {accuracy * 100:.2f}%")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_rice_classifier()
    sys.exit(0 if success else 1)
