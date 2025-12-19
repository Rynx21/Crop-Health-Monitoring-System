#!/usr/bin/env python3
"""
Download Required Crop Datasets from Original Sources

This script automates fetching classifier datasets for crops used by the app
from their original sources (primarily Kaggle), and organizes them into
archive_datasets/{crop}_classifier_dataset/train and val with an 80/20 split.

Supported crops: tomato, potato, chili, rice
Sources:
- PlantVillage (Kaggle: arjuntejaswi/plant-village) for tomato + chili
- PlantVillage Potato subset (Kaggle: aarishasifkhan/plantvillage-potato-disease-dataset)
- Rice Disease (Kaggle: tedmylo/ricerice-disease-image-dataset)

Usage examples:
  python download_datasets.py --all --dry-run
  python download_datasets.py --crops tomato potato --force

Notes:
  - Requires Kaggle API credentials in %USERPROFILE%\.kaggle\kaggle.json
  - Large datasets (e.g., rice) may take significant time and disk space
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).parent
ARCHIVE_DIR = BASE_DIR / "archive_datasets"


DATASET_SOURCES: Dict[str, Dict] = {
    # Tomato + Chili from PlantVillage
    "plantvillage": {
        "provider": "kaggle",
        "id": "arjuntejaswi/plant-village",
        "classes": {
            "tomato": [
                "Tomato___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites_Two_spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            ],
            "chili": [
                "Pepper__bell___healthy",
                "Pepper__bell___Bacterial_spot",
            ],
        },
    },

    # Potato subset (works reliably)
    "potato": {
        "provider": "kaggle",
        "id": "aarishasifkhan/plantvillage-potato-disease-dataset",
        "classes": {
            "potato": [
                "Potato___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
            ]
        },
    },

    # Rice diseases dataset
    "rice": {
        "provider": "kaggle",
        "id": "tedmylo/ricerice-disease-image-dataset",
        "classes": {
            "rice": [
                "BrownSpot",
                "Healthy",
                "Hispa",
                "LeafBlast",
            ]
        },
    },
}


def _ensure_kaggle_ready() -> bool:
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("Installing Kaggle API...")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        except Exception as e:
            print(f"! Failed to install Kaggle API: {e}")
            return False

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        print(f"! kaggle.json not found at {kaggle_json}")
        print("  Create from https://www.kaggle.com/settings/account and place it there.")
        return False
    return True


def kaggle_download(dataset_id: str, dest_dir: Path) -> Optional[Path]:
    """Download and unzip a Kaggle dataset into dest_dir. Returns root path."""
    try:
        import kaggle
        dest_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_id, path=str(dest_dir), unzip=False)
        # Find downloaded zip(s)
        zips = list(dest_dir.glob("*.zip"))
        if not zips:
            print(f"! No zip files found after download for {dataset_id}")
            return None
        root = dest_dir / "extracted"
        root.mkdir(exist_ok=True)
        for z in zips:
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(root)
            except zipfile.BadZipFile:
                print(f"! Bad zip file: {z.name}")
                return None
        return root
    except Exception as e:
        print(f"! Kaggle download failed for {dataset_id}: {e}")
        return None


def find_class_dirs(search_root: Path, target_names: List[str]) -> Dict[str, Path]:
    """Search recursively for folders whose name matches any target class name."""
    found: Dict[str, Path] = {}
    lower_targets = {t.lower(): t for t in target_names}
    for p in search_root.rglob("*"):
        if p.is_dir():
            name_lower = p.name.lower()
            if name_lower in lower_targets and lower_targets[name_lower] not in found:
                found[lower_targets[name_lower]] = p
    return found


def copy_split_class_images(src_class_dir: Path, dst_train: Path, dst_val: Path) -> None:
    imgs = [f for f in src_class_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    imgs.sort()
    if not imgs:
        return
    split = int(len(imgs) * 0.8)
    train_imgs = imgs[:split]
    val_imgs = imgs[split:]
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)
    for f in train_imgs:
        shutil.copy2(f, dst_train / f.name)
    for f in val_imgs:
        shutil.copy2(f, dst_val / f.name)


def organize_classifier_dataset(crop: str, class_dirs: Dict[str, Path]) -> Path:
    out_root = ARCHIVE_DIR / f"{crop}_classifier_dataset"
    for cls_name in class_dirs.keys():
        (out_root / "train" / cls_name).mkdir(parents=True, exist_ok=True)
        (out_root / "val" / cls_name).mkdir(parents=True, exist_ok=True)
    for cls_name, src_dir in class_dirs.items():
        copy_split_class_images(src_dir, out_root / "train" / cls_name, out_root / "val" / cls_name)
    return out_root


def download_for_crop(crop: str, force: bool = False) -> bool:
    # Determine which source block contains this crop
    source_key = None
    for key, info in DATASET_SOURCES.items():
        if crop in info.get("classes", {}):
            source_key = key
            break
    if not source_key:
        print(f"! No dataset source configured for crop: {crop}")
        return False

    expected_out = ARCHIVE_DIR / f"{crop}_classifier_dataset"
    if expected_out.exists() and not force:
        print(f"✓ Dataset already organized for {crop}: {expected_out}")
        return True

    source = DATASET_SOURCES[source_key]
    if source["provider"] != "kaggle":
        print(f"! Unsupported provider: {source['provider']}")
        return False

    if not _ensure_kaggle_ready():
        return False

    temp_dir = BASE_DIR / f"temp_{crop}_download"
    root = kaggle_download(source["id"], temp_dir)
    if not root:
        return False

    # Find classes in extracted content
    target_classes = source["classes"][crop]
    found = find_class_dirs(root, target_classes)
    missing = [c for c in target_classes if c not in found]
    if missing:
        print("! Missing class folders:")
        for m in missing:
            print(f"  - {m}")
        print(f"  Searched under: {root}")
        # Proceed with found ones only

    # Organize
    out_root = organize_classifier_dataset(crop, found)
    print(f"✓ Organized {crop} dataset at: {out_root}")

    # Cleanup temp
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and organize crop datasets from original sources")
    parser.add_argument("--all", action="store_true", help="Download for all supported crops")
    parser.add_argument("--crops", nargs="+", help="Specific crops to download (e.g., tomato potato)")
    parser.add_argument("--force", action="store_true", help="Redownload and re-organize even if exists")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without downloading")
    args = parser.parse_args()

    targets: List[str] = []
    if args.all:
        targets = ["tomato", "chili", "potato", "rice"]
    elif args.crops:
        targets = args.crops
    else:
        parser.print_help()
        return

    if args.dry_run:
        print("Planned downloads and outputs:")
        for crop in targets:
            # Determine source
            src_key = None
            for key, info in DATASET_SOURCES.items():
                if crop in info.get("classes", {}):
                    src_key = key
                    break
            if not src_key:
                print(f"- {crop}: NO SOURCE CONFIGURED")
                continue
            src = DATASET_SOURCES[src_key]
            out = ARCHIVE_DIR / f"{crop}_classifier_dataset"
            print(f"- {crop}: Kaggle {src['id']} -> {out}")
            print(f"  Classes: {', '.join(src['classes'][crop])}")
        return

    # Execute downloads
    success = True
    for crop in targets:
        ok = download_for_crop(crop, force=args.force)
        success = success and ok
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
