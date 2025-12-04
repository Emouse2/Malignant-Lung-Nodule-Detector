from pathlib import Path
import sys
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
import data_functions

def crop_nodule(slice_2d, xs, ys, padding=5):
    x_min, x_max = min(xs)-padding, max(xs)+padding
    y_min, y_max = min(ys)-padding, max(ys)+padding
    return slice_2d[y_min:y_max, x_min:x_max]