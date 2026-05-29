import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from src.dataset import CholecDataset

NUM_CLASSES = 13

dataset = CholecDataset()

data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

images, masks = next(iter(data_loader))
print(f"batch image shape : {images.shape}")
print(f"batch image dtype : {images.dtype}")
print(f"batch mask shape : {masks.shape}")
print(f"batch mask dtype : {masks.dtype}")
print(f"mask unique values :{torch.unique(masks)} ")

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

model.eval()
with torch.no_grad():
    outputs = model(images)
logits = outputs["out"]
print(f"logits dtype: {logits.dtype}")
print(f"logits shape : {logits.shape}")

criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
loss = criterion(logits, masks)

print(f"loss: {loss.item()}")
