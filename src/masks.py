from PIL import Image
import numpy as np

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
IGNORE_INDEX = 255


def decode_mask(mask_path):
    mask = Image.open(mask_path)
    arr = np.array(mask)
    mask_codes = arr[:, :, 0] # identical channels, we just need one
    decoded = np.full_like(mask_codes, fill_value=IGNORE_INDEX)

    for raw_value, class_id in LABEL_MAP.items():
        decoded[mask_codes == raw_value] = class_id
    return decoded

