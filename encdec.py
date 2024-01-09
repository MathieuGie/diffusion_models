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

batches = 128
epochs = 500


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=12)

    def forward(self, x):

        x = self.relu(self.conv1(x))#100
        #print(x.shape)
        x = self.relu(self.conv2(x))#50
        #print(x.shape)
        x = self.relu(self.conv3(x))#25
        #print(x.shape)
        x = self.relu(self.conv4(x))#12
        #print(x.shape)
        x = self.conv5(x)#1

        #print("end of encoder", x.shape)

        return self.sigmoid(x)
    
class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=12, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)


    def forward(self, x):

        x = self.relu(self.deconv1(x)) #12
        #print(x.shape)
        x = self.relu(self.deconv2(x))#25
        #print(x.shape)
        x = self.relu(self.deconv3(x))#50
        #print(x.shape)
        x = self.relu(self.deconv4(x))#100
        #print(x.shape)
        x = self.deconv5(x)#200

        #print("end of decoder", x.shape)

        return self.sigmoid(x)
    
class Encdec(pl.LightningModule):
    def __init__(self,):
        super().__init__()

        self.epochh=0

        self.encoder = Encoder().to(mps_device)
        self.decoder = Decoder().to(mps_device)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _step(self, batch, batch_idx):

        outputs = self(batch)
        #print("for loss:", outputs.shape, batch.shape)
        loss = nn.functional.mse_loss(outputs, batch)
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
        torch.save(self.encoder.state_dict(), "encoder.chkpt")
        torch.save(self.decoder.state_dict(), "decoder.chkpt")

        with torch.no_grad():

            image = test_dataset[0].to(mps_device)
            
            if not os.path.exists('predictions_encdec'):
                os.makedirs('predictions_encdec')

            image=self.encoder(image)
            image=self.decoder(image)

            pil_image = TF.to_pil_image(image)

            image_path = os.path.join('predictions_encdec', f"image_{self.epochh}.png")
            pil_image.save(image_path, inplace=True)
            

        self.epochh+=1
        self.train()
    

class ImageDataset(Dataset):
    def __init__(self, dir, transform=None, file_list=None, subset_fraction=1):

        self.directory = dir
        self.transform = transform

        self.all_images = []

        if file_list is None:
            file_list = []
            if os.path.exists(dir):
                file_list.extend(os.listdir(dir))

        if os.path.exists(dir):
            for img_name in file_list:
                if img_name in os.listdir(dir) and img_name[0]!=".":
                    img_path = os.path.join(dir, img_name)
                    self.all_images.append(img_path)

        self.all_images = random.sample(self.all_images, int(len(self.all_images) * subset_fraction))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        
        image = self.all_images[idx]
        image = Image.open(image).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed
])

dir = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/Diffusion/CAT_00_treated'

image_files = os.listdir(dir)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.3)  # Adjust test_size as needed
#print(len(train_files))
train_dataset = ImageDataset(dir, transform, file_list=train_files)
test_dataset = ImageDataset(dir, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batches, shuffle=True)

print("before mps device")
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

model = Encdec()
model.to(mps_device)

print("model init done")

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

trainer = pl.Trainer(max_epochs=epochs, accelerator="mps", logger=mlf_logger, log_every_n_steps=1)
trainer.fit(model, dataloader_train, dataloader_test)