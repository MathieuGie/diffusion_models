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


class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=25, padding=1)
        self.maxpolling1 = nn.MaxPool2d(padding=1, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=1)
        self.maxpolling2 = nn.MaxPool2d(padding=1, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=6, padding=1)

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=1025, out_channels=256, kernel_size=4, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)  
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1) 
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=3, padding=1)   
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=6, stride=4, padding=1)      

    def forward(self, x, t):
        # Encoding
        #print(x.shape)
        x = self.relu(self.conv1(x))
        #print(x.shape)
        x = self.maxpolling1(x)
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        x = self.maxpolling2(x)
        #print(x.shape)
        x = self.relu(self.conv3(x))

        # Concatenating with t
        #t_tensor = torch.full((x.size(0), 1, x.size(2), x.size(3)), t, device=x.device)
        #print(x.shape, t.shape)
        x = torch.cat((x, t), dim=1)
        #print(x.shape)

        # Decoding
        x = self.relu(self.deconv1(x))
        #print(x.shape)
        x = self.relu(self.deconv2(x))
        #print(x.shape)
        x = self.relu(self.deconv3(x))
        #print(x.shape)
        x = self.relu(self.deconv4(x))
        #print(x.shape)
        x = self.relu(self.deconv5(x))
        #print(x.shape)

        return x
    
    def _step(self, batch, batch_idx):

        # Basic function to compute the loss
        noisy_imgs, clean_imgs, t = batch
        outputs = self(noisy_imgs, t)
        loss = torch.nn.functional.mse_loss(outputs, clean_imgs)
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


class DenoisingDataset(Dataset):
    def __init__(self, base_dir, transform=None, file_list=None, subset_fraction=0.15):
        self.base_dir = base_dir
        self.transform = transform

        # List to store tuples of (more noisy image path, less noisy image path, noising step)
        self.image_pairs = []

        # If file_list is None, use all images in the directory
        if file_list is None:
            file_list = []
            for step in range(1, 11):
                step_dir = os.path.join(base_dir, f'step_{step}')
                if os.path.exists(step_dir):
                    file_list.extend(os.listdir(step_dir))

        # Iterate over all noising steps except the first one (step_0)
        for step in range(1, 11):
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
                            self.image_pairs.append((current_img_path, previous_img_path, step/10))

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

train_files, test_files = train_test_split(image_files, test_size=0.2)  # Adjust test_size as needed
#print(len(train_files))
train_dataset = DenoisingDataset(dir_all, transform, file_list=train_files)
test_dataset = DenoisingDataset(dir_all, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=True)

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

trainer = pl.Trainer(max_epochs=10, accelerator="mps", logger=mlf_logger, log_every_n_steps=1, val_check_interval=0.5)
trainer.fit(model, dataloader_train, dataloader_test)


#GENERATE NEW IMAGES

folder_name = "generated_images"
os.makedirs(folder_name, exist_ok=True)

image = torch.rand((3,100,100))
step=1

for i in range(10):

    batched_image = image.unsqueeze(0)

    step_tensor = torch.zeros((1,1,1,1))
    step_tensor[0,0,0,0]=step

    image = model(batched_image, step_tensor)[0,:,:,:]
    step-=1/10

    pil_image = TF.to_pil_image(image)

    image_path = os.path.join(folder_name, f"image_{i}.png")
    pil_image.save(image_path)
