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
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

epochs = 500
size = 20
batches = 128
noising_steps = 50


class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=4, padding=1, stride=2)
        #self.maxpolling1 = nn.MaxPool2d(padding=1, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=12, padding=1)
        #self.maxpolling2 = nn.MaxPool2d(padding=1, kernel_size=2)
        #self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5)

        # Decoder layers
        self.linear1 = nn.Linear(1025, 2000)
        #self.linear2 = nn.Linear(800, 500)
        self.linear3 = nn.Linear(2000, 3*size**2)

        self.epochh=0

    def forward(self, x, t):
        # Encoding
        #print(x.shape)
        x = self.relu(self.conv1(x))
        #print(x.shape)
        #x = self.maxpolling1(x)
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        #x = self.maxpolling2(x)
        #print(x.shape)
        #x = self.relu(self.conv3(x))

        # Concatenating with t
        #t_tensor = torch.full((x.size(0), 1, x.size(2), x.size(3)), t, device=x.device)
        #print(x.shape, t.shape)
        x = torch.cat((x, t), dim=1)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)
        # Decoding
        x = self.relu(self.linear1(x))
        #x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    
    def _step(self, batch, batch_idx):

        noisy_imgs, clean_imgs, t = batch

        outputs = self(noisy_imgs, t)

        target = noisy_imgs-clean_imgs
        target = target.view(target.size(0), -1)

        #target = torch.zeros(outputs.shape).to(mps_device)

        loss = nn.functional.mse_loss(outputs, target)
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
    
    def on_train_epoch_start(self):
        
        with torch.no_grad():

            image = torch.rand((3,size,size)).to(mps_device)
            step=1
            i=0
            
            if not os.path.exists('predictions'):
                os.makedirs('predictions', inplace=True)
            step_folder = os.path.join('predictions', f'epoch_{self.epochh}')

            os.makedirs(step_folder, exist_ok=True)

            for i in range(noising_steps):

                batched_image = image.unsqueeze(0)

                step_tensor = torch.zeros((1,1,1,1)).to(mps_device)
                step_tensor[0,0,0,0]=step

                result = model(batched_image, step_tensor)[0,:]
                result = result.view(3,size, size)
                image-=result.to(mps_device)

                step-=1/noising_steps

                pil_image = TF.to_pil_image(image)

                image_path = os.path.join(step_folder, f"image_{i}.png")
                pil_image.save(image_path, inplace=True)
                i+=1

        self.epochh+=1
        self.train()


class DenoisingDataset(Dataset):
    def __init__(self, base_dir, transform=None, file_list=None, subset_fraction=0.7):
        self.base_dir = base_dir
        self.transform = transform

        # List to store tuples of (more noisy image path, less noisy image path, noising step)
        self.image_pairs = []

        # If file_list is None, use all images in the directory
        if file_list is None:
            file_list = []
            for step in range(1, noising_steps+1):
                step_dir = os.path.join(base_dir, f'step_{step}')
                if os.path.exists(step_dir):
                    file_list.extend(os.listdir(step_dir))

        # Iterate over all noising steps except the first one (step_0)
        for step in range(1, noising_steps+1):
            print("step:", step)
            current_step_dir = os.path.join(base_dir, f'step_{step}')
            previous_step_dir = os.path.join(base_dir, f'step_{step-1}')

            # Check if both current and previous step directories exist
            if os.path.exists(current_step_dir) and os.path.exists(previous_step_dir):
                for img_name in file_list:
                    if img_name in os.listdir(current_step_dir):
                        current_img_path = os.path.join(current_step_dir, img_name)
                        previous_img_path = os.path.join(previous_step_dir, img_name)

                        # Check if the corresponding less noisy image exists in the previous step
                        if os.path.exists(previous_img_path):
                            self.image_pairs.append((current_img_path, previous_img_path, step/noising_steps))

        self.image_pairs = random.sample(self.image_pairs, int(len(self.image_pairs) * subset_fraction))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        more_noisy_img_path, less_noisy_img_path, step = self.image_pairs[idx]

        more_noisy_image = Image.open(more_noisy_img_path).convert('RGB')
        less_noisy_image = Image.open(less_noisy_img_path).convert('RGB')

        if self.transform:
            more_noisy_image = self.transform(more_noisy_image)
            less_noisy_image = self.transform(less_noisy_image)

        """
        pil_image = TF.to_pil_image(more_noisy_image)
        pil_image2 = TF.to_pil_image(less_noisy_image)

        # Plotting the images
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(pil_image)
        plt.title("More Noisy Image")
        plt.axis('off')  # To turn off axes for image plot

        plt.subplot(1, 2, 2)
        plt.imshow(pil_image2)
        plt.title("Less Noisy Image")
        plt.axis('off')

        plt.show()
        """

        step_tensor = torch.zeros((1,1,1))
        step_tensor[0,0,0]=step

        return more_noisy_image, less_noisy_image, step_tensor


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed
])


dir_images = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_0'
dir_all =  '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy'

image_files = os.listdir(dir_images)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.3)  # Adjust test_size as needed
#print(len(train_files))
train_dataset = DenoisingDataset(dir_all, transform, file_list=train_files)
test_dataset = DenoisingDataset(dir_all, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batches, shuffle=True)

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

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, dataloader_train, dataloader_test)





"""
#GENERATE NEW IMAGES

folder_name = "generated_images"
os.makedirs(folder_name, exist_ok=True)

image = torch.rand((3,size,size))
step=1

# Path to the 'step_20' folder
step_20_folder = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_noisy/step_20'  # Update this path to your actual folder path

# Get the list of image files in the folder
image_files = sorted([f for f in os.listdir(step_20_folder) if f.endswith('.jpg')])

# Load the first image
img_path = os.path.join(step_20_folder, image_files[0])
image = Image.open(img_path).convert('RGB')

transform = transforms.ToTensor()
image = transform(image)

for i in range(noising_steps):

    batched_image = image.unsqueeze(0)

    step_tensor = torch.zeros((1,1,1,1))
    step_tensor[0,0,0,0]=step

    result = model(batched_image, step_tensor)[0,:]
    result = result.view(3,size, size)
    image-=result

    step-=1/noising_steps

    pil_image = TF.to_pil_image(image)

    image_path = os.path.join(folder_name, f"image_{i}.png")
    pil_image.save(image_path, inplace=True)
"""