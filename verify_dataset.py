#!/usr/bin/env python3
"""
Verify the NIH ChestX-ray14 dataset structure and CSV file
"""

import os
import pandas as pd
from dataset import NIHDataset

def verify_dataset_structure():
    """Verify that the dataset structure is correct"""
    
    data_path = '/kaggle/input/data'
    csv_file = '/kaggle/input/data/Data_Entry_2017.csv'
    
    print("=" * 60)
    print("NIH ChestX-ray14 Dataset Verification")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    else:
        print(f"✓ CSV file found: {csv_file}")
    
    # Load CSV and show statistics
    df = pd.read_csv(csv_file)
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Total patients: {df['Patient ID'].nunique()}")
    
    # Check disease distribution
    print(f"\n🏥 Disease Label Distribution:")
    all_labels = []
    for labels in df['Finding Labels']:
        if labels != 'No Finding':
            all_labels.extend(labels.split('|'))
    
    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {label:20s}: {count:6d}")
    
    no_finding = (df['Finding Labels'] == 'No Finding').sum()
    print(f"  {'No Finding':20s}: {no_finding:6d}")
    
    # Check directory structure
    print(f"\n📁 Directory Structure:")
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        return False
    
    image_dirs = sorted([d for d in os.listdir(data_path) 
                        if d.startswith('images_') and os.path.isdir(os.path.join(data_path, d))])
    
    if not image_dirs:
        print(f"❌ No image directories found in {data_path}")
        return False
    
    print(f"  Found {len(image_dirs)} image directories:")
    total_images = 0
    for img_dir in image_dirs:
        full_path = os.path.join(data_path, img_dir, 'images')
        if os.path.exists(full_path):
            img_count = len([f for f in os.listdir(full_path) if f.endswith('.png')])
            total_images += img_count
            print(f"    {img_dir}/images: {img_count:6d} images")
        else:
            print(f"    ❌ {img_dir}/images not found")
    
    print(f"\n  Total images in directories: {total_images}")
    print(f"  Images in CSV: {len(df)}")
    
    # Test dataset loading
    print(f"\n🔧 Testing Dataset Loading:")
    try:
        # Create a small test split
        train_indices, val_indices, test_indices = NIHDataset.create_train_test_split(
            csv_file, test_size=0.15, val_size=0.1, random_state=42
        )
        
        print(f"\n✓ Train/Val/Test split created successfully")
        
        # Try loading a single sample
        from torchvision import transforms
        test_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ])
        
        test_dataset = NIHDataset(
            csv_file=csv_file,
            root_dir=data_path,
            transform=test_transform,
            indices=train_indices[:10]  # Just test first 10 samples
        )
        
        print(f"✓ Dataset created with {len(test_dataset)} samples")
        
        # Load first sample
        image, label = test_dataset[0]
        print(f"✓ Successfully loaded first sample:")
        print(f"    Image shape: {image.shape}")
        print(f"    Label shape: {label.shape}")
        print(f"    Number of positive labels: {label.sum().item()}")
        
        # Get image info
        info = test_dataset.get_image_info(0)
        print(f"    Image: {info['image_index']}")
        print(f"    Labels: {info['finding_labels']}")
        
        print(f"\n✅ All checks passed! Dataset is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_dataset_structure()
