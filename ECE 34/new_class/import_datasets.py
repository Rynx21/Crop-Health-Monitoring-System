#!/usr/bin/env python3
"""
Automatic Dataset Downloader & Importer
Downloads crop disease datasets and prepares them for training

Supported sources: Kaggle, Roboflow, local files
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import json

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("="*80)
    print("SETTING UP KAGGLE API")
    print("="*80)
    
    print("\n1. Go to https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. This downloads kaggle.json")
    print("4. Place it in: C:\\Users\\%USERNAME%\\.kaggle\\")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"\n✅ Found kaggle.json at {kaggle_json}")
        return True
    else:
        print(f"\n❌ kaggle.json not found at {kaggle_json}")
        print("Please download it and place in that location")
        return False

def install_kaggle():
    """Install Kaggle API"""
    print("Installing Kaggle API...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
        print("✅ Kaggle API installed")
        return True
    except Exception as e:
        print(f"❌ Failed to install Kaggle: {e}")
        return False

def download_from_kaggle(dataset_name, output_dir):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset path (e.g., 'username/dataset-name')
        output_dir: Where to save dataset
    """
    print(f"\n📥 Downloading from Kaggle: {dataset_name}")
    
    try:
        import kaggle
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        
        print(f"✅ Dataset downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def organize_detector_dataset(source_dir, crop_name):
    """
    Organize dataset for object detection training
    
    Expected source structure:
    - images/ folder with all images
    - annotations/ or labels/ folder with YOLO format .txt
    """
    print(f"\n🔄 Organizing detector dataset for {crop_name}...")
    
    base_dir = Path(f"{crop_name}_detector_dataset")
    os.makedirs(base_dir / "images" / "train", exist_ok=True)
    os.makedirs(base_dir / "images" / "val", exist_ok=True)
    os.makedirs(base_dir / "labels" / "train", exist_ok=True)
    os.makedirs(base_dir / "labels" / "val", exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Find images folder
    images_dir = None
    for root, dirs, files in os.walk(source_path):
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            images_dir = Path(root)
            break
    
    if not images_dir:
        print("❌ No images found in source directory")
        return False
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not image_files:
        print("❌ No image files found")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Split: 80% train, 20% val
    split_point = int(len(image_files) * 0.8)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    # Copy train images
    print(f"Copying {len(train_files)} training images...")
    for img in train_files:
        src = images_dir / img
        dst = base_dir / "images" / "train" / img
        shutil.copy2(src, dst)
        
        # Copy corresponding label if exists
        label_name = img.rsplit('.', 1)[0] + '.txt'
        label_src = images_dir.parent / "labels" / label_name
        if label_src.exists():
            label_dst = base_dir / "labels" / "train" / label_name
            shutil.copy2(label_src, label_dst)
    
    # Copy val images
    print(f"Copying {len(val_files)} validation images...")
    for img in val_files:
        src = images_dir / img
        dst = base_dir / "images" / "val" / img
        shutil.copy2(src, dst)
        
        # Copy corresponding label if exists
        label_name = img.rsplit('.', 1)[0] + '.txt'
        label_src = images_dir.parent / "labels" / label_name
        if label_src.exists():
            label_dst = base_dir / "labels" / "val" / label_name
            shutil.copy2(label_src, label_dst)
    
    print(f"✅ Detector dataset organized in {base_dir}")
    return True

def organize_classifier_dataset(source_dir, crop_name, classes):
    """
    Organize dataset for classification training
    
    Expected: Class subfolders with images
    or: Flat folder where filenames contain class info
    """
    print(f"\n🔄 Organizing classifier dataset for {crop_name}...")
    
    base_dir = Path(f"{crop_name}_classifier_dataset")
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in classes:
            os.makedirs(base_dir / split / class_name, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Check if source already has class subfolders
    class_dirs = [d for d in os.listdir(source_path) 
                  if os.path.isdir(source_path / d) and d.lower() in [c.lower() for c in classes]]
    
    if class_dirs:
        print(f"Found class folders: {class_dirs}")
        
        # Copy from existing class structure
        for class_dir in class_dirs:
            class_name = class_dir.lower()
            src_class = source_path / class_dir
            images = [f for f in os.listdir(src_class) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not images:
                continue
            
            # Split 80/20
            split_point = int(len(images) * 0.8)
            train_imgs = images[:split_point]
            val_imgs = images[split_point:]
            
            # Copy training images
            for img in train_imgs:
                src = src_class / img
                dst = base_dir / "train" / class_name / img
                shutil.copy2(src, dst)
            
            # Copy validation images
            for img in val_imgs:
                src = src_class / img
                dst = base_dir / "val" / class_name / img
                shutil.copy2(src, dst)
            
            print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")
    else:
        print("⚠️  Class folders not found, creating empty structure")
        print(f"Please manually organize images into:")
        for split in ['train', 'val']:
            for class_name in classes:
                print(f"  {base_dir}/{split}/{class_name}/")
    
    print(f"✅ Classifier dataset organized in {base_dir}")
    return True

def list_kaggle_datasets():
    """List popular crop disease datasets on Kaggle"""
    print("\n" + "="*80)
    print("POPULAR KAGGLE DATASETS FOR CROP DISEASES")
    print("="*80)
    
    datasets = {
        "Potato Disease": {
            "path": "vipomonozon/potato-disease-classification",
            "crops": ["potato"],
            "classes": 3,
            "images": 2152
        },
        "Corn Disease": {
            "path": "smarques/corn-or-maize-leaf-disease-classification",
            "crops": ["corn"],
            "classes": 4,
            "images": 4188
        },
        "Rice Disease": {
            "path": "tedmylo/ricerice-disease-image-dataset",
            "crops": ["rice"],
            "classes": 4,
            "images": 2500
        },
        "Tomato Disease": {
            "path": "arjuntejaswi/plant-village",
            "crops": ["tomato", "potato"],
            "classes": 39,
            "images": 54000
        },
        "Carrot & Vegetables": {
            "path": "search: 'carrot disease' or 'vegetable disease'",
            "crops": ["carrot", "lettuce"],
            "classes": "varies",
            "images": "varies"
        }
    }
    
    for name, info in datasets.items():
        print(f"\n📊 {name}")
        print(f"   Kaggle: {info['path']}")
        print(f"   Crops: {', '.join(info['crops'])}")
        print(f"   Classes: {info['classes']}")
        print(f"   Images: {info['images']}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and organize crop disease datasets')
    parser.add_argument('--list', action='store_true', help='List popular datasets')
    parser.add_argument('--setup-kaggle', action='store_true', help='Setup Kaggle API')
    parser.add_argument('--download', type=str, help='Download dataset (Kaggle path: user/dataset)')
    parser.add_argument('--crop', type=str, required=True, help='Crop name (carrot, potato, etc.)')
    parser.add_argument('--type', choices=['detector', 'classifier'], help='Dataset type')
    parser.add_argument('--source-dir', type=str, help='Local directory with dataset')
    parser.add_argument('--classes', nargs='+', help='Class names for classifier')
    
    args = parser.parse_args()
    
    if args.list:
        list_kaggle_datasets()
        return
    
    if args.setup_kaggle:
        setup_kaggle_api()
        install_kaggle()
        return
    
    if args.download:
        install_kaggle()
        setup_kaggle_api()
        temp_dir = f"temp_{args.crop}_download"
        download_from_kaggle(args.download, temp_dir)
        
        if args.type == 'detector':
            organize_detector_dataset(temp_dir, args.crop)
        elif args.type == 'classifier' and args.classes:
            organize_classifier_dataset(temp_dir, args.crop, args.classes)
    
    elif args.source_dir:
        if args.type == 'detector':
            organize_detector_dataset(args.source_dir, args.crop)
        elif args.type == 'classifier' and args.classes:
            organize_classifier_dataset(args.source_dir, args.crop, args.classes)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
