import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import datasets,transforms
from torchvision.utils import save_image
from PIL import Image
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import random
from sklearn.model_selection import train_test_split



class Encoder(pl.LightningModule):
    def __init__(self, m):
        super().__init__()
        
        layers = []
        in_channels = 3  # RGB images

        for _ in range(m):
            out_channels = in_channels * 2  # Double the channels after each layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # Adding batch normalization
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
  
    def forward(self, img):
        return self.conv_layers(img)

class Decoder(pl.LightningModule):
    def __init__(self, m):
        super().__init__()

        layers = []
        in_channels = 2 ** m * 3  # Depending on the encoder's last layer output channels

        for _ in range(m):
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # Adding batch normalization
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=3, stride=1, padding=1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, img_latent_space):
        return self.conv_layers(img_latent_space)
    

class Encoder_Decoder(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.encoder = Encoder(m=4)
        self.encoder.to(device)  
        self.decoder = Decoder(m=4)
        self.decoder.to(device)
        
    def forward(self,x):
        x=self.encoder(x)
        return self.decoder(x)
    
 
    #computation of the loss 
    def _step(self, batch, batch_idx):

        outputs = self(batch) #renvoie le resultat du decodeur 
        loss = torch.nn.functional.mse_loss(outputs, batch)
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


class image_dataset(Dataset):
    def __init__(self, dir, transform=None, file_list=None):
        self.dir = dir
        self.transform = transform
        if file_list is not None:
            self.images = file_list
        else:
            self.images = os.listdir(dir)  # Fallback to using all files if file_list is not provided

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image

# Chemin vers le dataset
dataset_path = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/Dataset/animals/cat'

# Transformations 
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensionnement des images
    transforms.ToTensor()
])


image_files = os.listdir(dataset_path)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.2)  # Adjust test_size as needed

train_dataset = image_dataset(dataset_path,transform, file_list=train_files)
test_dataset = image_dataset(dataset_path, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(test_dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


# Initialisation du modèle


model=Encoder_Decoder()
model.to(device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=50, logger=mlf_logger, log_every_n_steps=1, val_check_interval=0.1)

#trainer = pl.Trainer(max_epochs=50, accelerator="mps", logger=mlf_logger, log_every_n_steps=1, val_check_interval=0.1)
trainer.fit(model, dataloader_train, dataloader_val)