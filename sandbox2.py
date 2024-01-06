import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
import random
from sklearn.model_selection import train_test_split


# 1. Pytorch Lightning (=PL)
class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x, t):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)

        return x
    
    def _step(self, batch, batch_idx):

        # Basic function to compute the loss
        noisy_imgs, clean_imgs = batch
        outputs = self(noisy_imgs)
        loss = torch.nn.functional.mse_loss(outputs, clean_imgs)
        return loss

    
    def training_step(self,batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None, file_list=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        if file_list is not None:
            self.images = file_list
        else:
            self.images = os.listdir(noisy_dir)  # Fallback to using all files if file_list is not provided

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        noisy_img_name = os.path.join(self.noisy_dir, self.images[idx])
        clean_img_name = os.path.join(self.clean_dir, self.images[idx])

        noisy_image = Image.open(noisy_img_name).convert('RGB')
        clean_image = Image.open(clean_img_name).convert('RGB')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed
])


noisy_dir = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_1'
clean_dir = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_0'

image_files = os.listdir(clean_dir)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.2)  # Adjust test_size as needed

train_dataset = DenoisingDataset(noisy_dir, clean_dir, transform, file_list=train_files)
test_dataset = DenoisingDataset(noisy_dir, clean_dir, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=True)

#Devise:
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

model = CNN()
model.to(mps_device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=8, accelerator="mps", logger=mlf_logger, log_every_n_steps=1, val_check_interval=0.25)
trainer.fit(model, dataloader_train, dataloader_test)