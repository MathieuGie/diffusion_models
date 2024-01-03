import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import torch.nn as nn
import matplotlib.pyplot as plt


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


######## TEST #################

# Initialize your encoder and decoder
enc = encoder(m=1)  
dec = decoder(m=1)  

# Load your image
image_path = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/resize_animal/bear_0.jpg'  # replace with an actual image file name
image = Image.open(image_path).convert('RGB')


# Transform the image
transform = transforms.Compose([
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)  # Add batch dimension


# Pass the image through the encoder and decoder
with torch.no_grad():  # No need to track gradients for this
    encoded_img = enc(image)
    decoded_img = dec(encoded_img)

# Convert the tensor back to an image for visualization
output_image = transforms.ToPILImage()(decoded_img.squeeze(0))

# Plot the original and reconstructed image
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(0).permute(1, 2, 0))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title("Reconstructed Image")
plt.axis('off')

plt.show()
