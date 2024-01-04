import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

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
        return image
    

class ForwardDiffusion:

    def __init__(self, transform, input_path, output_folder, beta=0.1):

        self.transform=transform

        cat_dataset = CatImagesDataset(directory=input_path, transform=self.transform)
        self.data = DataLoader(cat_dataset, batch_size=32, shuffle=False)

        self.output_folder = output_folder
        self.beta = beta
        self.step = 1

        self.save_original_images(input_path, output_folder)

    def save_original_images(self, input_path, output_folder):
        step_0_folder = os.path.join(output_folder, 'step_0')
        os.makedirs(step_0_folder, exist_ok=True)
        cat_dataset = CatImagesDataset(directory=input_path, transform=self.transform)
        data_loader = DataLoader(cat_dataset, batch_size=32, shuffle=False)
        for n_batch, batch in enumerate(data_loader, start=1):
            for i, image in enumerate(batch):
                image = image.clamp(0, 1)
                save_image(image, os.path.join(step_0_folder, f'image_{n_batch-1}-{i}.jpg'))

    def add_noise(self, inputs):
        noise = torch.randn_like(inputs) * np.sqrt(self.beta)
        return np.sqrt(1-self.beta)* inputs + noise

    def save_images(self, images, step, n_batch):
        step_folder = os.path.join(self.output_folder, f'step_{step}')
        os.makedirs(step_folder, exist_ok=True)
        for i, image in enumerate(images):
            image = image.clamp(0, 1)
            save_image(image, os.path.join(step_folder, f'image_{n_batch-1}-{i}.jpg'))

    def run(self, n_times):
        for _ in range(n_times):
            i=0
            for batch in self.data:
                #print(batch)
                i+=1
                noisy_batch = self.add_noise(batch)
                self.save_images(noisy_batch, self.step, i)

            input_path = os.path.join(self.output_folder, f'step_{self.step}')
            dataset = CatImagesDataset(directory=input_path, transform=self.transform)
            self.data = DataLoader(dataset, batch_size=32, shuffle=False)
            print("done step", self.step)
            self.step += 1

# Define your transforms here
transformations = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
])


input_path =  'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/resize_animal'
output_path = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/noisy_animal'

forward_diff = ForwardDiffusion(transformations, input_path, output_path)
<<<<<<< HEAD
forward_diff.run(30)

=======
forward_diff.run(10)
>>>>>>> 035f8c03075141f12593721eff3531a0b270d6be
