import torch

from src.metrics import dice, foreground_iou, mean_iou, per_class_iou


preds = torch.tensor([
    [0, 1],
    [1, 2],
])

targets = torch.tensor([
    [0, 1],
    [2, 2],
])

per_class_iou_result = per_class_iou(preds, targets, num_classes=3)
mean_iou_result = mean_iou(preds, targets, num_classes=3)
foreground_iou_result = foreground_iou(preds, targets, num_classes=3)
dice_result = dice(preds, targets, num_classes=3)

print(f"per-class IoU result: {per_class_iou_result}")
print("per-class IoU expected: tensor([1.0000, 0.5000, 0.5000])")
print("------------")
print(f"mIoU result: {mean_iou_result}")
print("mIoU expected: tensor(0.6667)")
print("------------")
print(f"foreground IoU result: {foreground_iou_result}")
print("foreground IoU expected: tensor(0.5000)")
print("------------")
print(f"Dice result: {dice_result}")
print("Dice expected: tensor([1.0000, 0.6667, 0.6667])")
