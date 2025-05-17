import os, time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

from UNET.model import build_unet
from UNET.utils import create_dir, seeding

def mask_parse(mask):
    """ Convert single channel binary mask to 3 channels for visualization """
    mask = np.expand_dims(mask, axis=-1)    # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Paths """
    input_images_dir = "CombinedData/cataract/retinal_images"
    output_masks_dir = "CombinedData/cataract/masked_images"
    os.makedirs(output_masks_dir, exist_ok=True)
    test_x = sorted(glob(os.path.join(input_images_dir, "*.*")))
    H = 512
    W = 512
    for i, img_path in enumerate(test_x):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (W, H))
        cv2.imwrite(img_path, img_resized)
    
    
    """ Load images """
    test_x = sorted(glob(os.path.join(input_images_dir, "*.*")))
    """ Resize images to 512x512 """
    
    
    """ Hyperparameters """
    H = 512
    W = 512
    checkpoint_path = "files/checkpoint.pth"

    """ Load the model checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for i, x_path in tqdm(enumerate(test_x), total=len(test_x)):
        """ Extract the image name """
        name = os.path.basename(x_path).split(".")[0]
        # mask_path = os.path.join(output_masks_dir, f"{name}.png")
        # if os.path.exists(mask_path):
        #     #print(f"Mask for {name} already exists. Skipping.")
        #     continue

        """ Reading and preprocessing the input image """
        image = cv2.imread(x_path, cv2.IMREAD_COLOR)  # (512, 512, 3)
        x = np.transpose(image, (2, 0, 1))            # (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)                 # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x).to(device)

        with torch.no_grad():
            """ Prediction """
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)       # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Save the generated mask """
        mask_image = pred_y * 255  # Rescale to [0, 255]
        mask_path = os.path.join(output_masks_dir, f"{name}.png")
        cv2.imwrite(mask_path, mask_image)

    print("Generated masks saved in the 'masked_images' folder.")
