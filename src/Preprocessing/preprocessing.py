from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import pandas as pd
import os
import ast
from sklearn.model_selection import train_test_split
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
        self.labels_csv = labels_csv
        self.base_folder = base_folder
        self.transform = transform
    
    def __len__(self):
        df = pd.read_csv(self.labels_csv)
        return len(df)
    
    def __getitem__(self, heading_in, heading_data):
        target_data_get = data_functions.heading_to_heading(
            self.labels_csv,
            heading_in,
            ["patient_id", "xs", "ys", "z-slice", "malignancy"],
            heading_data
        )
        data_get = target_data_get.tolist()
        if len(data_get) > 1:
            print("headings not specific enough")
        data_get = target_data_get.tolist()[0]
        patient_id = data_get[0]
        xs = ast.literal_eval(data_get[1])
        ys = ast.literal_eval(data_get[2])
        z_slice = float(data_get[3])
        malignancy = int(data_get[4])

        series_path = data_functions.get_ct_folder(
            self.labels_csv,
            self.base_folder,
            patient_id
        )
        
        slices, volume = data_functions.load_ct_series(series_path)
        z_index = data_functions.get_slice_index(slices, z_slice)
        slice_img = volume[z_index]
        slice_img = preprocessing_functions.crop_nodule(slice_img, xs, ys)
        
        image_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(malignancy, dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label

dataset = LungNoduleDataset(labels_path, images_path)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

df = pd.read_csv(labels_path)
patients = df["patient_id"].unique()
train_patients, temp_patients = train_test_split(
    patients, test_size=0.30, random_state=42
)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.50, random_state=42
)
train_df = df[df["patient_id"].isin(train_patients)]
val_df = df[df["patient_id"].isin(val_patients)]
test_df = df[df["patient_id"].isin(test_patients)]

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))