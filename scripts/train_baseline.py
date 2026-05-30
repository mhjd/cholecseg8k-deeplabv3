import torch
import time
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from src.dataset import CholecDataset
from src.metrics import dice, foreground_iou, mean_iou, per_class_iou

NUM_CLASSES = 13
RESOLUTION = (427, 240)
TRAIN_VIDEO_IDS = [1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37]
VAL_VIDEO_IDS = [43, 48]
BATCH_SIZE = 4

train_dataset = CholecDataset(image_size=RESOLUTION, video_ids=TRAIN_VIDEO_IDS)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CholecDataset(image_size=RESOLUTION, video_ids=VAL_VIDEO_IDS)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

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
            images = images.to(device)
            masks = masks.to(device)
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


def print_epoch_progres(epoch_start, batch_start, batch_end, batch_done, n_batches):
        epoch_percentage_progress = batch_done / n_batches * 100
        batch_time_elapsed = batch_end - batch_start
        epoch_time_elapsed = batch_end - epoch_start
        
        projected_total_instant = batch_time_elapsed * n_batches
        instant_based_remaining_time = projected_total_instant - epoch_time_elapsed

        avg_batch_time = epoch_time_elapsed / batch_done
        projected_total_avg = avg_batch_time * n_batches
        avg_based_remaining_time = avg_batch_time * (n_batches - batch_done)

        print(f"{indent(2)}===== Batch info (size {BATCH_SIZE}) =====")
        print(f"{indent(2)}Epoch progress... {epoch_percentage_progress:.4f}%")
        print(f"{indent(2)}Current batch time : {batch_time_elapsed:.2f}s")
        print(f"{indent(2)}Projected total epoch time at current speed : {projected_total_instant:.2f}s")
        print(f"{indent(2)}Projected total epoch time with average: {projected_total_avg:.2f}s")
        # print(f"{indent(2)}Projected epoch remaining time at current speed : {instant_based_remaining_time:.2f}s")
        # print(f"{indent(2)}Projected epoch remaining time with average: {avg_based_remaining_time:.2f}s")
        print(f"{indent(2)}=====================")
        
def train_one_epoch(model, data_loader):
    model.train()
    loss_acc = 0.0
    n_batches= len(data_loader)
    epoch_start = time.perf_counter()
    for idx, (images, masks) in enumerate(data_loader):
        batch_start = time.perf_counter()
        # training step
        optimizer.zero_grad()
        images = images.to(device)
        masks = masks.to(device)
        
        
        outputs = model(images)
        logits = outputs["out"]
        loss_train = criterion(logits, masks)
        loss_train.backward()
        optimizer.step()
        loss_acc += loss_train.item()
        
        
        batch_end = time.perf_counter()
        batch_done = idx + 1
        print_epoch_progres(epoch_start, batch_start, batch_end, batch_done, n_batches)
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

