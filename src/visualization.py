import numpy as np
from PIL import Image

OUTPUT_DIR = "outputs/"
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


def image_tensor_to_numpy(image):
    image = image.detach().cpu()
    image = image.permute(1, 2, 0)
    image = image.numpy()
    return (image * 255).astype(np.uint8)


def mask_to_numpy(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return mask


def visualize_mask(mask):
    mask = mask_to_numpy(mask)
    visualized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in MAPPING_CODE_TO_COLOR.items():
        visualized_mask[class_id == mask] = color
    return visualized_mask

def save_sample_visualization(image, mask, output_name):
    image = image_tensor_to_numpy(image)
    visualized_mask = visualize_mask(mask)
    
    mask_img = Image.fromarray(visualized_mask)
    mask_img.save(OUTPUT_DIR + output_name + "_mask.png")
    
    saving_image = Image.fromarray(image) 
    saving_image.save(OUTPUT_DIR + output_name + ".png")


def save_prediction_visualization(image, ground_truth_mask, pred_mask, output_name):
    image = image_tensor_to_numpy(image)
    visualized_ground_truth = visualize_mask(ground_truth_mask)
    visualized_pred_mask = visualize_mask(pred_mask)

    Image.fromarray(image).save(OUTPUT_DIR + output_name + ".png")
    Image.fromarray(visualized_ground_truth).save(OUTPUT_DIR + output_name + "_mask.png")
    Image.fromarray(visualized_pred_mask).save(OUTPUT_DIR + output_name + "_mask_pred.png")

    panel = np.concatenate([image, visualized_ground_truth, visualized_pred_mask], axis=1)
    Image.fromarray(panel).save(OUTPUT_DIR + output_name + "_panel.png")
    
