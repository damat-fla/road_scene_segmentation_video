from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

class CamVidDataset(Dataset):

    # The __init__ method initializes the dataset with the paths to the images and 
    # masks folders, and an optional transform to apply to the images and masks.
    def __init__(self, images_folder, masks_folder, image_transform=None, mask_transform=None):
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        # Apply different transformations to images and masks if provided.
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Order the images and masks by their filenames, this ensures that the 
        # images and masks are paired correctly. Ignore hidden files.
        self.images = sorted([f for f in os.listdir(images_folder) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(masks_folder) if not f.startswith('.')])

    def __len__(self):
        return len(self.images)
    
    # The __getitem__ method retrieves an image and its corresponding mask by index.
    # It applies the specified transformations to both the image and mask if provided.
    # Then it returns the transformed image and mask as a tuple.
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.images[idx])
        mask_path = os.path.join(self.masks_folder, self.masks[idx])

        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # shape (height, width, channels)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.image_transform:
            image = Image.fromarray(image) # shape (width, height)
            image = self.image_transform(image) # shape (channels, height, width)
        else:
            image = transforms.ToTensor()(image) # shape (channels, height, width)

        if self.mask_transform:
            mask = Image.fromarray(mask)
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        return image, mask
