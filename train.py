import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from models.unet import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.camvid_dataset import CamVidDataset
from utils.helper import get_device, save_checkpoint, compute_validation_loss


device = get_device()
N_EPOCHS = 10
BATCH_SIZE = 8
checkpoint_dir = './checkpoints'
save_every = 2  # Save every 2 epochs

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
print(f"Trainable params: {trainable_params} \n")



# ========== Loss and Optimizer ==========
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ========== Training Loop ==========

best_val_loss = np.inf
val_loss = np.inf

for epoch in range(1, N_EPOCHS+1):

    print(f"================================    Starting epoch {epoch}/{N_EPOCHS}      ========================")

    model.train()
    running_loss = 0

    # Using tqdm for progress bar and showing loss
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}", leave=False)
    
    for images, masks in loop: # shape (BxCxHxW)
        images = images.to(device) 
        masks = masks.to(device)

        optimizer.zero_grad() # no gradient accumulation
        preds = model(images) # forward pass

        loss = criterion(preds, masks) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update parameters

        running_loss += loss.item() * images.size(0) # loss.item() returns the mean loss over the batch

        loop.set_postfix(loss=loss.item(), val_loss=val_loss) # update the progress bar with the current loss

    epoch_loss = running_loss / len(train_loader.dataset) # average loss over the entire dataset
    
    # Compute validation loss
    val_loss = compute_validation_loss(model, val_loader, criterion, device)
    
    # Check whether the val_loss improved
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
        best_val_loss = val_loss
        is_best = True
    else:
        print(f"Validation loss did not improve: {val_loss:.4f} (best: {best_val_loss:.4f})")
        is_best = False


    # Saving the checkpoint
    save_checkpoint(
        model_name='unet',
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=epoch_loss,
        val_loss=val_loss if is_best else None, # Save also the best checkpoint every time the validation loss improves
        save_every_n=save_every  # Save every 2 epoch. If None, save only the last model and the best.
    )

    print(f"Epoch {epoch}/{N_EPOCHS}, Loss: {epoch_loss:.4f}, Val_Loss = {val_loss:.4f} \n")
print("Training complete.")
