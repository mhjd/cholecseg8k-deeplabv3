import numpy as np

LABEL_MAP = {
    50 : 0, # Black background
    11 : 1, # Abdonimal wall
    21 : 2, # Liver
    13 : 3, # Gastrointestinal tract
    12 : 4, # Fat
    31 : 5, # Grasper
    23 : 6, # Connective tissue
    24 : 7, # Blood
    25 : 8, # Cystic duct
    32 : 9, # L-hook Electrocautery
    22 : 10, # Gallbladder
    33 : 11, # Hepatic vein
    5  : 12, # Liver ligament
}
IGNORE_INDEX = 255


def decode_mask(mask_array):
    arr = np.array(mask_array)
    mask_codes = arr[:, :, 0] # identical channels, we just need one
    decoded = np.full_like(mask_codes, fill_value=IGNORE_INDEX)

    for raw_value, class_id in LABEL_MAP.items():
        decoded[mask_codes == raw_value] = class_id
    return decoded

