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


class NoisyCleanDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.images = os.listdir(noisy_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.images[idx]))
        clean_img = Image.open(os.path.join(self.clean_dir, self.images[idx]))

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img

transform = transforms.Compose([
    transforms.ToTensor(),
    # Ajoutez d'autres transformations si nécessaire
])

train_dataset = NoisyCleanDataset("G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/noisy_animals", "G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/resize_animal"
, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = DenoisingCNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    for data in train_loader:
        noisy_imgs, clean_imgs = data
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

