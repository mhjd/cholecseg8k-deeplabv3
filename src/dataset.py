from PIL import Image
import numpy as np
import torch
from pathlib import Path
from src.masks import decode_mask
from torch.utils.data import Dataset

def collect_pair_images_masks():
    dataset_path = Path("dataset/CholecSeg8k")
    mask_paths = sorted(list(dataset_path.rglob("*_endo_watershed_mask.png")))
    pairs_images_masks = []
    for mask_path in mask_paths:
        #print(f"mask path : {mask_path}")
        image_name = mask_path.name.replace("_watershed_mask", "")
        image_path = mask_path.with_name(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for mask {mask_path}. Expected image path: {image_path}")
        pairs_images_masks.append((image_path, mask_path))
    return pairs_images_masks

def get_video_id(path):
    for part in path.parts:
        if part.startswith("video"):
            number_text = part.removeprefix("video")
            if number_text.isdigit():
                return part, int(number_text)
    raise ValueError(f"Could not find video id in path: {path}")

def collect_pair_images_masks_by_video_id(video_id):
    if video_id < 10:
        video_id_str = '0' + str(video_id)
    else:
        video_id_str = str(video_id)
    dataset_path = Path(f"dataset/CholecSeg8k/video{video_id_str}")
    mask_paths = sorted(list(dataset_path.rglob("*_endo_watershed_mask.png")))
    pairs_images_masks = []
    for mask_path in mask_paths:
        image_name = mask_path.name.replace("_watershed_mask", "")
        image_path = mask_path.with_name(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for mask {mask_path}. Expected image path: {image_path}")
        
        pairs_images_masks.append((image_path, mask_path))
    return pairs_images_masks
        

def collect_pair_images_masks_by_video_ids(video_ids):
    pairs_images_masks = []
    for video_id in video_ids:
        pairs_images_masks.extend(collect_pair_images_masks_by_video_id(video_id))
    return pairs_images_masks
        
class CholecDataset(Dataset):
    # PIL uses (width, height)
    def __init__(self, image_size=None, video_ids=None):
        
        self.image_size = image_size
        if video_ids is not None:
            self.pair_images_masks = collect_pair_images_masks_by_video_ids(video_ids)
        else:
            self.pair_images_masks = collect_pair_images_masks()

    def __len__(self):
        return len(self.pair_images_masks)

    def __getitem__(self, idx):
        img = Image.open(self.pair_images_masks[idx][0]).convert("RGB")
        mask = Image.open(self.pair_images_masks[idx][1]).convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, resample=Image.Resampling.BILINEAR)
            mask = mask.resize(self.image_size, resample=Image.Resampling.NEAREST)

        img_arr = np.array(img)
        mask_arr = np.array(mask)
        decoded_mask_arr = decode_mask(mask_arr)

        image_tensor = torch.from_numpy(img_arr)
        # H, W, C -> C, H, W
        # permutation because pytorch doesn't store images the same way than numpy
        image_tensor = image_tensor.permute(2, 0, 1)
        # normalisation
        image_tensor = image_tensor.float() / 255.0
        decoded_mask_tensor = torch.from_numpy(decoded_mask_arr).long() # long/i64 because it's classes

        return image_tensor, decoded_mask_tensor
        
        
