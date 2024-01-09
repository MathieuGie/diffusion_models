import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

T=100
beta_max = 0.5

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
    

class ForwardDiffusion:

    def __init__(self, transform, input_path, output_folder, T, beta_max):

        self.transform=transform

        cat_dataset = CatImagesDataset(directory=input_path, transform=self.transform)
        self.data = DataLoader(cat_dataset, batch_size=32, shuffle=False)

        self.output_folder = output_folder
        self.beta = 0
        self.step = 1

        self.T = T
        self.beta_max = beta_max

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
                save_image(image, os.path.join(step_0_folder, filename))

    def add_noise(self, inputs):
        noise = torch.randn(inputs.shape) * np.sqrt(self.beta)
        return np.clip(np.sqrt(1-self.beta)* inputs + noise, 0, 1)
    
    def update_beta(self, step):
        self.beta = self.beta_max/self.T *step +0.0001

    def save_images(self, images, filenames, step):
        step_folder = os.path.join(self.output_folder, f'step_{step}')
        os.makedirs(step_folder, exist_ok=True)
        for image, filename in zip(images, filenames):  # Use the filename
            image = image.clamp(0, 1)
            save_image(image, os.path.join(step_folder, filename))

    def run(self, n_times):
        for step in range(n_times):
            self.update_beta(step+1)
            print(self.beta)

            for i, (batch, filenames) in enumerate(self.data):  # Unpack images and filenames
                noisy_batch = self.add_noise(batch)
                self.save_images(noisy_batch, filenames, self.step)
                i += 1

            # Prepare for the next step
            input_path = os.path.join(self.output_folder, f'step_{self.step}')
            dataset = CatImagesDataset(directory=input_path, transform=self.transform)
            self.data = DataLoader(dataset, batch_size=32, shuffle=False)
            print("done step", self.step)
            self.step += 1

# Define your transforms here
transformations = transforms.Compose([
    transforms.Resize((20, 20)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
])


input_path = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_treated'
output_path = '../CAT_00_noisy'

forward_diff = ForwardDiffusion(transformations, input_path, output_path, T, beta_max)
forward_diff.run(T)