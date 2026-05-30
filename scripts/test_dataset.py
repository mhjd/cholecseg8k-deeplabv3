import numpy as np
from src.dataset import CholecDataset
import torch

dataset = CholecDataset(image_size=(427, 240))

print(f"Size dataset: {len(dataset)}")
def check_dataset_value(idx):
    element = dataset[idx]
    its_img = element[0]
    its_mask = element[1]
    print(f"img dtype : {its_img.dtype}")
    print(f"img shape: {its_img.shape}")
    print(f"mask dtype : {its_mask.dtype}")
    print(f"mask shape: {its_mask.shape}")
    print(f"img min/max : {its_img.min()} / {its_img.max()}")
    print(f"mask unique values : {torch.unique(its_mask)}")
    print("------------")
check_dataset_value(0)
check_dataset_value(100)
check_dataset_value(1000)
check_dataset_value(8000)
