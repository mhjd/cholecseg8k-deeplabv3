import torch


def per_class_iou(preds, targets, num_classes, ignore_index=255):
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    
    ious = []
    nan = torch.tensor(float("nan"), device=preds.device)
    for class_id in range(num_classes):
        preds_is_class = preds == class_id
        target_is_class = targets == class_id

        intersection = (preds_is_class & target_is_class).sum()
        union = (preds_is_class | target_is_class).sum()
        if union == 0:
            iou = nan
        else:
            iou = intersection.float() / union.float()
        ious.append(iou)
    return torch.stack(ious)

def mean_iou(preds, targets, num_classes, ignore_index=255):
    ious = per_class_iou(preds, targets, num_classes, ignore_index=ignore_index)
    return torch.nanmean(ious)

def foreground_iou(preds, targets, num_classes, ignore_index=255):
    ious = per_class_iou(preds, targets, num_classes, ignore_index=ignore_index)
    return torch.nanmean(ious[1:]) # first class is black background

def dice(preds, targets, num_classes, ignore_index=255):
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    
    scores = []
    nan = torch.tensor(float("nan"), device=preds.device)
    for class_id in range(num_classes):
        preds_is_class = preds == class_id
        target_is_class = targets == class_id
        
        intersection = (preds_is_class & target_is_class).sum()
        add_prediction_target = preds_is_class.sum() + target_is_class.sum()
        if add_prediction_target == 0:
            score = nan
        else:
            score = 2.0 * intersection.float() / add_prediction_target.float()
        scores.append(score)
    return torch.stack(scores)
