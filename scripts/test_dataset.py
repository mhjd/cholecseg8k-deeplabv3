import numpy as np
from src.dataset import CholecDataset

dataset = CholecDataset()

print(f"Size dataset: {len(dataset)}")
def check_dataset_value(idx):
    element = dataset[idx]
    its_img = element[0]
    its_mask = element[1]
    print(f"dataset[{idx}] img shape : {its_img.shape}")
    print(f"dataset[{idx}] mask shape : {its_mask.shape}")
    print(f"img unique values : {np.unique(its_img)}")
    print(f"mask unique values : {np.unique(its_mask)}")
    print("------------")
check_dataset_value(0)
check_dataset_value(100)
check_dataset_value(1000)
check_dataset_value(8000)
