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


####### CONVOLUTIONAL MODEL ##################
class Encoder(nn.Module):
    def __init__(self, m):
        super(Encoder, self).__init__()
        
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

class Decoder(nn.Module):
    def __init__(self, m):
        super(Decoder, self).__init__()

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

########## TRAINING ########################

# Chemin vers le dataset
dataset_path = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/Dataset/animals'

# Transformations 
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensionnement des images
    transforms.ToTensor()
])

# Charger le dataset
full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# Séparer le dataset en ensembles d'entraînement et de validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


# Initialisation du modèle
encoder = Encoder(m=4)
encoder.to(device)  
decoder = Decoder(m=4)
decoder.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

from tqdm import tqdm

def train_model(encoder, decoder, criterion, optimizer, train_loader, val_loader, epochs=25):
    for epoch in range(epochs):
        encoder.train()  # Set encoder to training mode
        decoder.train()  # Set decoder to training mode

        total_train_loss = 0.0

        # Training loop with progress bar
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Training]')
        for inputs, _ in train_progress_bar:
            inputs = inputs.to(device)
            
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            encoded_imgs = encoder(inputs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss calculation
            loss = criterion(decoded_imgs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            train_progress_bar.set_postfix(loss=loss.item())

        # Average training loss for the epoch
        average_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        encoder.eval()  # Set encoder to evaluation mode
        decoder.eval()  # Set decoder to evaluation mode
        total_val_loss = 0.0

        # Validation loop with progress bar
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Validation]')
        with torch.no_grad():
            for inputs, _ in val_progress_bar:
                inputs = inputs.to(device)

                # Forward pass
                encoded_imgs = encoder(inputs)
                decoded_imgs = decoder(encoded_imgs)

                # Loss calculation
                loss = criterion(decoded_imgs, inputs)
                total_val_loss += loss.item() * inputs.size(0)
                val_progress_bar.set_postfix(loss=loss.item())

        # Average validation loss for the epoch
        average_val_loss = total_val_loss / len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}')

    return encoder, decoder

trained_encoder, trained_decoder = train_model(encoder, decoder, criterion, optimizer, train_loader, val_loader, epochs=25)
