import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split


class NIHDataset(torch.utils.data.Dataset):
    """
    NIH ChestX-ray14 Dataset
    Multi-label classification for 14 thoracic diseases
    """
    
    # 14 disease labels in NIH ChestX-ray14 dataset
    LABELS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia'
    ]
    
    def __init__(self, csv_file, root_dir, transform=None, indices=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations (Data_Entry_2017.csv)
            root_dir (string): Directory containing image subdirectories (images_001, images_002, etc.)
            transform (callable, optional): Optional transform to be applied on a sample
            indices (list, optional): Specific indices to use (for train/test split)
        """
        self.df = pd.read_csv(csv_file)
        
        # If indices provided, filter the dataframe
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self.LABELS
        self.num_classes = len(self.LABELS)
        
        # Build image path mapping for the multi-directory structure
        self._build_image_path_map()
        
    def _build_image_path_map(self):
        """
        Build a mapping from image filename to full path
        Handles the structure: data/images_XXX/images/*.png
        """
        self.image_path_map = {}
        
        # Find all image subdirectories (images_001, images_002, etc.)
        image_dirs = glob.glob(os.path.join(self.root_dir, 'images_*', 'images'))
        
        # Build mapping from filename to full path
        for img_dir in image_dirs:
            for img_file in os.listdir(img_dir):
                if img_file.endswith('.png'):
                    self.image_path_map[img_file] = os.path.join(img_dir, img_file)
        
        print(f"Found {len(self.image_path_map)} images across {len(image_dirs)} directories")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image file name
        img_filename = self.df.iloc[idx, 0]
        
        # Use the path mapping to find the full path
        if img_filename in self.image_path_map:
            img_path = self.image_path_map[img_filename]
        else:
            raise FileNotFoundError(f"Image {img_filename} not found in any subdirectory")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get finding labels (second column)
        finding_labels = self.df.iloc[idx, 1]
        
        # Create multi-label binary vector
        label_vector = np.zeros(self.num_classes, dtype=np.float32)
        
        # Parse the finding labels (separated by '|')
        if finding_labels != 'No Finding':
            diseases = finding_labels.split('|')
            for disease in diseases:
                if disease in self.labels:
                    label_idx = self.labels.index(disease)
                    label_vector[label_idx] = 1.0
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Convert label vector to tensor
        label_vector = torch.from_numpy(label_vector)
        
        return image, label_vector
    
    def get_labels_for_idx(self, idx):
        """
        Get the list of disease labels for a specific index
        """
        finding_labels = self.df.iloc[idx, 1]
        if finding_labels == 'No Finding':
            return []
        return finding_labels.split('|')
    
    def get_image_info(self, idx):
        """
        Get additional information about the image
        """
        row = self.df.iloc[idx]
        return {
            'image_index': row[0],
            'finding_labels': row[1],
            'follow_up': row[2],
            'patient_id': row[3],
            'patient_age': row[4],
            'patient_gender': row[5],
            'view_position': row[6]
        }
    
    @staticmethod
    def create_train_test_split(csv_file, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/validation/test splits based on patient ID to avoid data leakage
        
        Args:
            csv_file: Path to Data_Entry_2017.csv
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            train_indices, val_indices, test_indices
        """
        df = pd.read_csv(csv_file)
        
        # Split by patient ID to avoid data leakage
        unique_patients = df['Patient ID'].unique()
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Get indices for each split
        train_indices = df[df['Patient ID'].isin(train_patients)].index.tolist()
        val_indices = df[df['Patient ID'].isin(val_patients)].index.tolist()
        test_indices = df[df['Patient ID'].isin(test_patients)].index.tolist()
        
        print(f"Split summary:")
        print(f"  Train: {len(train_indices)} images from {len(train_patients)} patients")
        print(f"  Val:   {len(val_indices)} images from {len(val_patients)} patients")
        print(f"  Test:  {len(test_indices)} images from {len(test_patients)} patients")
        
        return train_indices, val_indices, test_indices