import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from src.dataset import CholecDataset
from src.metrics import dice, foreground_iou, mean_iou, per_class_iou

NUM_CLASSES = 13
TRAIN_VIDEO_IDS = [1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37]
VAL_VIDEO_IDS = [43, 48]

train_dataset = CholecDataset(image_size=(106, 60), video_ids=TRAIN_VIDEO_IDS)
train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = CholecDataset(image_size=(106, 60), video_ids=VAL_VIDEO_IDS)
val_data_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def indent(n):
    return n*"      "
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
    print(f"{indent(2)}===== Metrics =====")
    print(f"{indent(2)}mIoU: {miou_acc / n_batches}")
    print(f"{indent(2)}foreground IoU: {foreground_iou_acc / n_batches}")
    print(f"{indent(2)}Dice: {dice_acc / n_batches}")
    print(f"{indent(2)}===================")
def train_one_epoch(model, data_loader):
    model.train()
    loss_acc = 0.0
    number_images = len(data_loader)
    for idx, (images, masks) in enumerate(data_loader):
        # training step
        optimizer.zero_grad()
        
        outputs = model(images)
        logits = outputs["out"]
        # preds = logits.argmax(dim=1)
        loss_train = criterion(logits, masks)
        loss_train.backward()
        optimizer.step()
        loss_acc += loss_train.item()
        print(f"{indent(1)}Epoch progress... {idx/number_images*100}%")
    loss = loss_acc / len(data_loader)
    print(f"{indent(1)}average loss : {loss}")
    return model
def train_n_epoch(model, data_loader, n, val_data_loader):
    for i in range(n):
        print(f"===== Epoch {i + 1} =====")
        model = train_one_epoch(model, data_loader)
        evaluate_one_epoch(model, val_data_loader)
        print("=====================")
    return model
    

train_n_epoch(model, train_data_loader, 1, val_data_loader)

