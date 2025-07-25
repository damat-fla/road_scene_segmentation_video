'''This script contains helper functions for orchestrating the entire process'''
import os
import torch
import numpy as np

# A function to determine the device to be used for training
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# A function to compute validation loss
def compute_validation_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            batch_loss = criterion(preds, masks)
            val_loss += batch_loss.item()*imgs.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss

# This function always saves the last checkpoint, and optionally saves the best 
# checkpoint (according to validation loss) and saves checkpoints every "save_every_n" 
# epochs if specified.
def save_checkpoint(
    model_name,
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    loss: float = None,
    val_loss: float = None,
    save_every_n: int = None
):
    # Create the directory if it doesn't exist
    model_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    # Prepare the dictionary to be saved
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if loss is not None:
        checkpoint['loss'] = loss

    # Save the best checkpoint if the loss has improved
    if val_loss is not None:
        best_path = os.path.join(model_path, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"New best checkpoint saved: {best_path}")

    # Save every N epochs (if requested)
    if save_every_n is not None and epoch % save_every_n == 0:
        epoch_path = os.path.join(model_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"Saving epoch {epoch} checkpoint to: {epoch_path}")

    # Always save the last checkpoint - always overwrite the last checkpoint
    last_path = os.path.join(model_path, 'checkpoint_last.pth')
    torch.save(checkpoint, last_path)
    print(f"Saving last checkpoint to: {last_path}")




