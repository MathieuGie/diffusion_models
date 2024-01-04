import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import os

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

class PairedImagesDataset(Dataset):
    def __init__(self, dir_step_1, dir_step_2, transform=None):
        self.dir_step_1 = dir_step_1
        self.dir_step_2 = dir_step_2
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(dir_step_1) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path_1 = os.path.join(self.dir_step_1, file_name)
        img_path_2 = os.path.join(self.dir_step_2, file_name)

        image_t_1 = Image.open(img_path_1).convert('RGB')
        image_t = Image.open(img_path_2).convert('RGB')

        if self.transform:
            image_t_1 = self.transform(image_t_1)
            image_t = self.transform(image_t)

        return image_t, image_t_1


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)  # No activation function in the final layer

        return x


class DiffusionProcess():
    ## Need a function to find the epsilon_t
    ## Need a nn to find epsilon_theta
    ## Need a funnction to make diference and backprop

    def __init__(self, CNN, batch_size, dir_step_1, dir_step_2, transform):

        self.cnn = CNN
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.001)

        self.batch_size = batch_size

        self.dataset = PairedImagesDataset(dir_step_1, dir_step_2, transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


    def eps_t(self, image_t, image_t_1):
        return image_t-image_t_1
    
    def eps_theta (self, image):
        return self.cnn(image)
    
    def get_loss(self, image_t, image_t_1):
        target = self.eps_t(image_t, image_t_1)
        prediction = self.eps_theta(image_t)

        L = self.loss(target, prediction)

        return L
    
    def training(self, num_epochs):
        loss_history = []

        for epoch in range(num_epochs):
            print("epoch", epoch)
            i=0
            for image_t, image_t_1 in self.data_loader:

                L = self.get_loss(image_t, image_t_1)

                print("batch done", i)

                L/=self.batch_size

                self.optimizer.zero_grad()
                L = self.get_loss(image_t, image_t_1)
                L.backward()

                self.optimizer.step()

                if i%1==0:  # Save the plot for every 10 epochs

                    loss_history.append(L.item())

                    plt.figure()
                    plt.plot(loss_history)

                    plt.title('Loss')
                    
                    # Save the figure to the same file location every time
                    plt.savefig('loss.png')
                    
                    # Close the figure to free memory
                    plt.close()

                    i=0

                i+=1
    


transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

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

#Running:
cnn = CNN()
#cnn.to(mps_device)

noisy_path = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_1'
not_noisy_path = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_0'
diffusion_process = DiffusionProcess(cnn, batch_size=32, dir_step_1=noisy_path, dir_step_2=not_noisy_path, transform=transformations)
diffusion_process.training(num_epochs=1000)



