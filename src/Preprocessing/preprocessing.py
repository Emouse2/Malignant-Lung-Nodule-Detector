from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

src_path = Path(__file__).parent.parent
functions_path = src_path / "Dataset"
sys.path.append(str(functions_path))
import data_functions # type: ignore
import preprocessing_functions # type: ignore

data_path = src_path.parent / "Data"
labels_path = data_path / "labels.csv"
images_path = data_path / "LIDC-IDRI"

class LungNoduleDataset(Dataset):
    def __init__(self, labels_csv, base_folder, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.base_folder = base_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["patient_id"]
        series_number = row["series_number"]
        series_path = os.path.join(self.base_folder, patient_id, f"CT{series_number:03d}")
        
        volume = data_functions.load_ct_series(series_path)
        z_index = data_functions.get_slice_index(
            [float(z) for z in row["slice_z"].split(",")],
            float(row["z-slice"])
        )
        slice_img = volume[z_index]
        
        xs = [int(x) for x in row["xs"].split(",")]
        ys = [int(y) for y in row["ys"].split(",")]
        slice_img = preprocessing_functions.crop_nodule(slice_img, xs, ys)
        
        image_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(row["malignancy"]), dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label

dataset = LungNoduleDataset(labels_path, "Dataset")
loader = DataLoader(dataset, batch_size=8, shuffle=True)