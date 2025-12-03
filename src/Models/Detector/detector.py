from pathlib import Path
import sys
import torch
from torch import nn

current_dir = Path(__file__).parent
data_functions_path = current_dir.parent.parent / "Dataset"
sys.path.append(str(data_functions_path))
import data_functions # type: ignore

train_functions_path = current_dir.parent
sys.path.append(str(train_functions_path))
import train_functions

data_path = current_dir.parent.parent / "Data"
labels = data_path / "labels.csv"
images_folder = data_path / "LIDC-IDRI"

class DetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        return x

torch.manual_seed(42)
model = DetectorModel()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)