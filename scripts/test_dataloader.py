from src.dataset import CholecDataset
from torch.utils.data import DataLoader
import torch

dataset = CholecDataset()

data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

first_batch = next(iter(data_loader))
print(f"batch image shape : {first_batch[0].shape}")
print(f"batch image dtype : {first_batch[0].dtype}")
print(f"batch mask shape : {first_batch[1].shape}")
print(f"batch mask dtype : {first_batch[1].dtype}")
print(f"mask unique values :{torch.unique(first_batch[1])} ")
