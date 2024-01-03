import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self,m):
        super(encoder,self).__init__()
        
        layers = []
        in_channels = 3  # we have RGB images


        #The encoder uses convolutional layers to progressively reduce the spatial dimensions while increasing the feature depth, 
        #compressing the image into a denser form.
        for _ in range(m):
            out_channels = in_channels * 2  # Example to double the channels after each layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
  
    def forward(self,img):

        img_latent_space = self.conv_layers(img)
        return img_latent_space



class decoder(nn.Module):
    def __init__(self,m):
        super(decoder,self).__init__()

        layers = []
        in_channels = 2 ** m * 3  # Adjust this depending on the encoder's last layer output channels

        for _ in range(m):
            out_channels = in_channels // 2  # Halve the number of channels in each layer
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final layer to produce the output with the original number of channels (e.g., 3 for RGB images)
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=3, stride=1, padding=1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self,img_latent_space):

        img=self.conv_layers(img_latent_space)
        return img




