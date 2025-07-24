import torch
from torch import nn
from tqdm import tqdm
from models.unet import UNet
from torchvision import transforms
from utils.helper import get_device
from torch.utils.data import DataLoader
from datasets.camvid_dataset import CamVidDataset


device = get_device()
N_EPOCHS = 3
BATCH_SIZE = 8

# ========== Dataset and DataLoader ==========
w, h = 720, 960 # width and height of the images in the dataset

image_transform = transforms.Compose([
    transforms.Resize((w // 2, h // 2)), # resize to half the original size
    transforms.ToTensor() # converts to tensor and scales pixel values to [0, 1] range
])

label_transform = transforms.Compose([
    transforms.Resize((w // 2, h // 2), 
                      interpolation=transforms.InterpolationMode.NEAREST 
                      ), # use nearest interpolation to avoid introducing new classes
    transforms.ToTensor()
])

train_dataset = CamVidDataset(
    images_folder='./data/CamVid/train/imgs',
    masks_folder='./data/CamVid/train/labels',
    image_transform=image_transform,
    mask_transform=label_transform
)

val_dataset = CamVidDataset(
    images_folder='./data/CamVid/val/imgs',
    masks_folder='./data/CamVid/val/labels',
    image_transform=image_transform,
    mask_transform=label_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



# ========== Model instantiation ==========
model = UNet(in_channels=3, out_channels=3, base_channels=16)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")



# ========== Loss and Optimizer ==========
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ========== Training Loop ==========
for epoch in range(N_EPOCHS):
    
    model.train()
    running_loss = 0

    # Using tqdm for progress bar and showing loss
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False)
    
    for images, masks in loop: # shape (BxCxHxW)
        images = images.to(device) 
        masks = masks.to(device)

        optimizer.zero_grad() # no gradient accumulation
        preds = model(images) # forward pass

        loss = criterion(preds, masks) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update parameters

        running_loss += loss.item() * images.size(0) # loss.item() returns the mean loss over the batch

        loop.set_postfix(loss=loss.item()) # update the progress bar with the current loss

    epoch_loss = running_loss / len(train_loader.dataset) # average loss over the entire dataset
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss:.4f}")
print("Training complete.")
