import torch
from src.dataset import CholecDataset
from src.visualization import save_prediction_visualization
from torchvision.models.segmentation import deeplabv3_resnet50


NUM_CLASSES = 13
TEST_VIDEO_IDS = [52, 55]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = deeplabv3_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

state_dict = torch.load("outputs/best_deeplabv3_resnet50.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()



dataset_52 = CholecDataset(image_size=(427, 240), video_ids=[52])
dataset_55 = CholecDataset(image_size=(427, 240), video_ids=[55])

def visualize_one_dataset_elem(dataset, img_name, image_index_list):
    
    for image_index in image_index_list:
        image, mask = dataset[image_index]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_batch)
            pred_mask = output["out"].argmax(dim=1)
            pred_mask = pred_mask[0]

        save_prediction_visualization(image, mask, pred_mask, img_name + "_" + str(image_index))

visualize_one_dataset_elem(dataset_52, "video_52", [0, 100, 400, 700])
visualize_one_dataset_elem(dataset_55, "video_55", [0, 100, 200])
