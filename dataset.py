# dataset.py
from torch.utils.data import Dataset
import pandas as pd
import torch
import cv2
from utils import get_transform

class CustomDataset(Dataset):
    def __init__(self, image_paths, image_CLAHE_paths, labels, feature_csv_path):
        self.image_paths = image_paths
        self.image_CLAHE_paths = image_CLAHE_paths
        self.labels = labels

        features_df = pd.read_csv(feature_csv_path)
        self.features_dict = {
            row['image_path']: [row['idm'], row['entropy'], row['contrast']]
            for _, row in features_df.iterrows()
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        original_image_path = self.image_paths[idx]
        clahe_image_path = self.image_CLAHE_paths[idx]

        clahe_image = cv2.imread(clahe_image_path)
        texture_features = self.features_dict[original_image_path]

        transform = get_transform()
        clahe_image = transform(clahe_image)

        return clahe_image, torch.tensor(texture_features, dtype=torch.float32), self.labels[idx]
