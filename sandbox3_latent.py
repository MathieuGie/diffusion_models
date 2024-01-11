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
from encdec import Decoder
import time

epochs = 800
size = 40
batches = 128
noising_steps = 25


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.epochh = 0

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.linear1 = nn.Linear(513, 4100)
        self.linear2 = nn.Linear(4100, 2050)
        self.linear3 = nn.Linear(2050, 1024)
        self.linear4 = nn.Linear(1024, 512)

        self.decoder = Decoder()
        self.decoder.load_state_dict(torch.load("decoder.chkpt"))
        self.decoder.eval()

    def forward(self, x, t):

        x = x.view(x.size(0), -1)
        t = t.view(t.size(0), -1)

        x = torch.cat((x, t), dim=1)

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)

        return x
    
    def _step(self, batch, batch_idx):

        noisy_imgs, clean_imgs, t = batch

        #print("hoho", noisy_imgs.shape, clean_imgs.shape, t.shape )

        outputs = self(noisy_imgs, t)

        target = noisy_imgs-clean_imgs
        #print("1", target.shape)
        target = target.view(target.size(0), -1)
        #print("2", target.shape)
        #print("std", torch.std(target), torch.std(outputs))
        #print("target", torch.max(target), torch.min(target))
        #print("output", torch.max(outputs), torch.min(outputs))

        loss = nn.functional.mse_loss(outputs, target)

        #print("LOSS", loss)
        #time.sleep(1)
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

            image = torch.rand((512,1)).to(mps_device)
            step=1
            i=0
            
            if not os.path.exists('predictions_latent'):
                os.makedirs('predictions_latent')
            step_folder = os.path.join('predictions_latent', f'epoch_{self.epochh}')

            os.makedirs(step_folder, exist_ok=True)

            for i in range(noising_steps):

                batched_image = image.unsqueeze(0)

                step_tensor = torch.zeros((1,1)).to(mps_device)
                step_tensor[0,0]=step

                result = model(batched_image, step_tensor)[0,:]

                result = result.view(result.size(0), 1)

                print("result", torch.mean(result), torch.std(result))

                image-=result.to(mps_device)

                dec_image = self.decoder(image.reshape(1, image.shape[0], 1, 1))

                step-=1/noising_steps

                pil_image = TF.to_pil_image(dec_image.reshape(dec_image.shape[1], size, size))

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

        more_noisy_image = torch.load(more_noisy_img_path)
        less_noisy_image = torch.load(less_noisy_img_path)

        step_tensor = torch.zeros((1)) 
        step_tensor[0]=step

        #print(more_noisy_image.shape,less_noisy_image.shape, step_tensor.shape )

        return more_noisy_image, less_noisy_image, step_tensor


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed
])


dir_images = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_latent_noisy/step_0'
dir_all =  '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_latent_noisy'

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

model = MLP()
model.to(mps_device)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, dataloader_train, dataloader_test)

