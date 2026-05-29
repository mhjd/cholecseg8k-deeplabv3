import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from src.dataset import CholecDataset

NUM_CLASSES = 13

dataset = CholecDataset()
subset_dataset = Subset(dataset, range(6))

data_loader = DataLoader(subset_dataset, batch_size=2, shuffle=False)

images, masks = next(iter(data_loader))
print(f"batch image shape : {images.shape}")
print(f"batch image dtype : {images.dtype}")
print(f"batch mask shape : {masks.shape}")
print(f"batch mask dtype : {masks.dtype}")
print(f"mask unique values :{torch.unique(masks)} ")

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


model.eval()
with torch.no_grad():
    outputs = model(images)
    logits = outputs["out"]
    loss_before = criterion(logits, masks)
print(f"logits dtype: {logits.dtype}")
print(f"logits shape : {logits.shape}")
print(f"loss before : {loss_before.item()}")

model.train()
# overfit loop
for i in range(10):
    # training loop
    
    loss_acc = 0.0
    for images, masks in data_loader:
        # training step
        optimizer.zero_grad()
        
        outputs = model(images)
        logits = outputs["out"]
        loss_train = criterion(logits, masks)
        loss_train.backward()
        optimizer.step()
        loss_acc += loss_train.item()
    loss = loss_acc / len(data_loader)
    print(f"average training loss of epoch {i + 1}: {loss}")
