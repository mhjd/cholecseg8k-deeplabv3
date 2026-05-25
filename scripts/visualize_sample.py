import numpy as np
from PIL import Image
from src.dataset import CholecDataset

dataset = CholecDataset()

MAPPING_CODE_TO_COLOR = {
    0:  (0, 114, 178),    # blue
    1:  (213, 94, 0),     # reddish orange
    2:  (0, 158, 115),    # green
    3:  (204, 121, 167),  # pink
    4:  (230, 159, 0),    # orange
    5:  (86, 180, 233),   # light blue
    6:  (128, 0, 128),    # purple
    7:  (128, 64, 0),     # brown
    8:  (240, 228, 66),   # yellow
    9:  (0, 139, 139),    # dark blue
    10: (220, 20, 60),    # red
    11: (107, 142, 35),   # olive green
    12: (70, 70, 70),     # gray
    255: (255, 255, 255)  # white
}

output_dir = "outputs/"

def create_visualization(idx, output_image_name):
    image = dataset[idx][0]
    mask = dataset[idx][1]

    visualized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in MAPPING_CODE_TO_COLOR.items():
        visualized_mask[class_id == mask] = color
        
    mask_img = Image.fromarray(visualized_mask)
    mask_img.save(output_dir + output_image_name + "_mask.png")

    saving_image = Image.fromarray(image)
    saving_image.save(output_dir + output_image_name + ".png")


create_visualization(1, "first_image")
