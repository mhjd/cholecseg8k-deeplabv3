import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from src.dataset import CholecDataset
from src.metrics import dice, foreground_iou, mean_iou, per_class_iou

NUM_CLASSES = 13

dataset = CholecDataset(image_size=(427, 240))
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



def evaluate_one_epoch(model, data_loader):
    model.eval()
    miou_acc = 0.0
    foreground_iou_acc = 0.0
    dice_acc = 0.0
    with torch.no_grad():
        for images, masks in data_loader:
            outputs = model(images)
            preds = outputs["out"].argmax(dim=1)
            miou_acc += mean_iou(preds, masks, NUM_CLASSES).item()
            foreground_iou_acc += foreground_iou(preds, masks, NUM_CLASSES).item()
            dice_acc += torch.nanmean(dice(preds, masks, NUM_CLASSES)).item()
    n_batches = len(data_loader)
    print("===== Metrics =====")
    print(f"mIoU: {miou_acc / n_batches}")
    print(f"foreground IoU: {foreground_iou_acc / n_batches}")
    print(f"Dice: {dice_acc / n_batches}")
    print("===================")

evaluate_one_epoch(model, data_loader)
# overfit
for i in range(10):
    # training loop
    model.train()
    loss_acc = 0.0
    for images, masks in data_loader:
        # training step
        optimizer.zero_grad()
        
        outputs = model(images)
        logits = outputs["out"]
        preds = logits.argmax(dim=1)
        loss_train = criterion(logits, masks)
        loss_train.backward()
        optimizer.step()
        loss_acc += loss_train.item()
    loss = loss_acc / len(data_loader)
    print(f"average training loss of epoch {i + 1}: {loss}")
    
evaluate_one_epoch(model, data_loader)
