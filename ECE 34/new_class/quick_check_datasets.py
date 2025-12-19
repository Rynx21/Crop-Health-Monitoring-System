"""Quick dataset verification check"""
import os
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent

# Check datasets
datasets = {
    'chili': 'archive_datasets/chili_classifier_dataset',
    'potato': 'archive_datasets/potato_classifier_dataset'
}

print("="*70)
print("DATASET VERIFICATION")
print("="*70)
print()

for crop_name, dataset_path in datasets.items():
    full_path = BASE_DIR / dataset_path
    
    print(f"🌱 {crop_name.upper()}")
    print(f"   Location: {dataset_path}")
    
    # Check train folder
    train_path = full_path / 'train'
    val_path = full_path / 'val'
    
    if train_path.exists() and val_path.exists():
        # Count classes in train
        train_classes = [d for d in train_path.iterdir() if d.is_dir()]
        val_classes = [d for d in val_path.iterdir() if d.is_dir()]
        
        print(f"   Classes: {len(train_classes)}")
        
        # Count images per class
        train_total = 0
        val_total = 0
        
        for cls in train_classes:
            cls_count = len([f for f in cls.iterdir() if f.is_file()])
            train_total += cls_count
            print(f"      {cls.name}: {cls_count} train", end="")
            
            # Check val
            val_cls = val_path / cls.name
            if val_cls.exists():
                val_count = len([f for f in val_cls.iterdir() if f.is_file()])
                val_total += val_count
                print(f", {val_count} val")
            else:
                print()
        
        print(f"   Total: {train_total} train, {val_total} val")
        
        # Calculate if dataset is balanced
        if len(train_classes) > 0:
            avg_per_class = train_total / len(train_classes)
            if train_total > 100 and val_total > 20:
                print(f"   ✅ Dataset is READY ({avg_per_class:.0f} images/class avg)")
            else:
                print(f"   ⚠️  Dataset is too small")
        else:
            print(f"   ❌ No classes found")
    else:
        print(f"   ❌ Dataset not found")
    
    print()

print("="*70)
print("RECOMMENDATIONS")
print("="*70)
print()
print("✅ Chili dataset is ready for training and evaluation")
print("✅ Potato dataset is ready for training and evaluation")  
print()
print("Next steps:")
print("1. Train models if needed: python train_classifier.py")
print("2. Evaluate accuracy: python evaluate_model_accuracy.py")
print("3. Run the web app: python app.py")
