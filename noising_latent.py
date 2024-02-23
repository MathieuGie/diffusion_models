import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
from encdec import Encoder

T=100
beta_max = 0.02

class CatImagesDataset(Dataset):

    #To transform the images 
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]
    
class CatTensorsDataset(Dataset):

    #To transform the images 
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = torch.load(img_path)
        return image, self.image_files[idx]
    

class ForwardDiffusion:

    def __init__(self, transform, input_path, output_folder, T, beta_max):

        self.transform=transform

        cat_dataset = CatImagesDataset(directory=input_path, transform=self.transform)
        self.data = DataLoader(cat_dataset, batch_size=32, shuffle=False)

        self.output_folder = output_folder
        self.beta = 0.0001
        self.step = 1

        self.T = T
        self.beta_max = beta_max

        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load("encoder.chkpt"))
        self.encoder.eval()

        self.save_original_images(input_path, output_folder)


    def save_original_images(self, input_path, output_folder):
        step_0_folder = os.path.join(output_folder, 'step_0')
        os.makedirs(step_0_folder, exist_ok=True)
        cat_dataset = CatImagesDataset(directory=input_path, transform=self.transform)
        data_loader = DataLoader(cat_dataset, batch_size=32, shuffle=False)
        for batch in data_loader:
            for i in range(batch[0].shape[0]):  # Correctly unpack the image and filename from each tuple in the batch

                image, filename = batch[0][i], batch[1][i]
                image = image.clamp(0, 1)

                image = torch.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

                enc_image = self.encoder.forward(image)
                enc_image = enc_image.view(1, -1)
                print(enc_image.shape)

                #print(enc_image.shape)
                torch.save(enc_image, os.path.join(step_0_folder, filename))

        input_path = os.path.join(self.output_folder, f'step_0')
        dataset = CatTensorsDataset(directory=input_path, transform=self.transform)
        self.data = DataLoader(dataset, batch_size=32, shuffle=False)
                

    def add_noise(self, inputs):
        noise = torch.randn(inputs.shape) * np.sqrt(self.beta)
        #print(torch.mean(noise), torch.std(noise))
        return torch.clip(np.sqrt(1-self.beta)* inputs + noise, 0, 1)
    
    def update_beta(self, step):
        self.beta =  np.clip(self.beta_max/self.T *step +0.0001, 0, 1)

    def save_images(self, images, filenames, step):
        step_folder = os.path.join(self.output_folder, f'step_{step}')
        os.makedirs(step_folder, exist_ok=True)
        for image, filename in zip(images, filenames):  # Use the filename

            if isinstance(image, torch.Tensor) is False:
                image = image.clamp(0, 1)
            
                if self.transform:
                    image = self.transform(image)

            #enc_image = self.encoder.forward(image)
            torch.save(image, os.path.join(step_folder, filename))


    def run(self, n_times):
        for step in range(n_times):
            self.update_beta(step)
            print(self.beta)

            for i, (batch, filenames) in enumerate(self.data):  # Unpack images and filenames

                #print(torch.std(batch), torch.max(batch), torch.min(batch))
                noisy_batch = self.add_noise(batch)
                #print(filenames)
                self.save_images(noisy_batch, filenames, self.step)
                i += 1

            # Prepare for the next step
            input_path = os.path.join(self.output_folder, f'step_{self.step}')
            dataset = CatTensorsDataset(directory=input_path, transform=self.transform)
            self.data = DataLoader(dataset, batch_size=32, shuffle=False)
            print("done step", self.step)
            self.step += 1

# Define your transforms here
transformations = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
])

input_path = '/Users/mathieugierski/Nextcloud/Macbook M3/Diffusion/CAT_00_treated'
output_path = '../CAT_00_latent_noisy'

forward_diff = ForwardDiffusion(transformations, input_path, output_path, T, beta_max)
forward_diff.run(T)