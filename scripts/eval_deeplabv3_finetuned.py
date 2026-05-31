import torch
from torch.utils.data import DataLoader
from src.dataset import CholecDataset
from src.device import get_device
from src.metrics import mean_iou, dice, foreground_iou
from torchvision.models.segmentation import deeplabv3_resnet50

NUM_CLASSES = 13
BATCH_SIZE = 4
TEST_VIDEO_IDS = [52, 55]

device = get_device()

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

state_dict = torch.load("outputs/best_deeplabv3_resnet50.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

eval_dataset = CholecDataset(image_size=(427, 240), video_ids=TEST_VIDEO_IDS)
eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)


miou_acc = 0.0
foreground_iou_acc = 0.0
dice_acc = 0.0
with torch.no_grad():
       for images, masks in eval_data_loader:
           images = images.to(device)
           masks = masks.to(device)
           outputs = model(images)
           preds = outputs["out"].argmax(dim=1)
           miou_acc += mean_iou(preds, masks, NUM_CLASSES).item()
           foreground_iou_acc += foreground_iou(preds, masks, NUM_CLASSES).item()
           dice_acc += torch.nanmean(dice(preds, masks, NUM_CLASSES)).item()
n_batches = len(eval_data_loader)

avg_miou = miou_acc / n_batches
avg_fg = foreground_iou_acc / n_batches
avg_dice = dice_acc / n_batches
print(f"===== Metrics =====")
print(f"mIoU: {avg_miou}")
print(f"foreground IoU: {avg_fg}")
print(f"Dice: {avg_dice}")
print(f"===================")
