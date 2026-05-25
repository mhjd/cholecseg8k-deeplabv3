from PIL import Image
from pathlib import Path
import numpy as np

FOLDER = "dataset/CholecSeg8k/video01/video01_00080/"
IMAGE = "frame_100_endo_watershed_mask.png"
LABEL_MAP = {
    50 : 0,
    11 : 1,
    21 : 2,
    13 : 3,
    12 : 4,
    31 : 5,
    23 : 6,
    24 : 7,
    25 : 8,
    32 : 9,
    22 : 10,
    33 : 11,
    5  : 12,
}

IGNORE_INDEX = 255 # not a class, seems to be borders

allowed_codes = set(LABEL_MAP.keys()) | {IGNORE_INDEX, 0} # we add 0, explained in README.md



def decode_img(img_path):
    img = Image.open(img_path)
    arr = np.array(img)
    mask_codes = arr[:, :, 0] # identical channels, we just need one
    decoded = np.full_like(mask_codes, fill_value=IGNORE_INDEX)

    for raw_value, class_id in LABEL_MAP.items():
        decoded[mask_codes == raw_value] = class_id
    return decoded

img_path = FOLDER + IMAGE
def find_unknown_codes(img_path):
    img = Image.open(img_path)
    arr = np.array(img)
    mask_codes = arr[:, :, 0] # identical channels, we just need one

    raw_codes = np.unique(mask_codes)
    unknown_codes = [code for code in raw_codes if code not in allowed_codes]
    return unknown_codes

dataset_path = Path("dataset/CholecSeg8k")
mask_paths = list(dataset_path.rglob("*_endo_watershed_mask.png"))

for mask_path in mask_paths:
    unknown_codes = find_unknown_codes(mask_path)
    if len(unknown_codes) > 0:
        print(f"Mask path : {mask_path}, unknown_codes : {unknown_codes}")
        # never reached, because the dataset is as excepted (except 255 and 0)
print(f" Unknown codes : {find_unknown_codes(img_path)}")
