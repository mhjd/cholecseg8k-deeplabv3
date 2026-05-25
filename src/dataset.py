from PIL import Image
import numpy as np
from pathlib import Path
from src.masks import decode_mask
from torch.utils.data import Dataset

def collect_pair_image_mask():
    dataset_path = Path("dataset/CholecSeg8k")
    mask_paths = sorted(list(dataset_path.rglob("*_endo_watershed_mask.png")))
    pairs_image_mask = []
    for mask_path in mask_paths:
        #print(f"mask path : {mask_path}")
        image_name = mask_path.name.replace("_watershed_mask", "")
        image_path = mask_path.with_name(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for mask {mask_path}. Expected image path: {image_path}")
        pairs_image_mask.append((image_path, mask_path))
    return pairs_image_mask

class CholecDataset(Dataset):
    def __init__(self):
        self.pair_image_mask = collect_pair_image_mask()

    def __len__(self):
        return len(self.pair_image_mask)

    def __getitem__(self, idx):
        img = Image.open(self.pair_image_mask[idx][0])
        mask = Image.open(self.pair_image_mask[idx][1])

        img_arr = np.array(img)
        mask_arr = np.array(mask)
        decoded_mask_arr = decode_mask(mask_arr)
        return img_arr, decoded_mask_arr
        
        
