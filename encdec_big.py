import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
import random
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

batches = 128
epochs = 1
PARALLEL_TRAINING = True



a=64
b=512
c=4096
d=512
e=122#
f=256#
g=512#

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=a, kernel_size=4, padding=1, stride=2)
        self.norm1 = nn.BatchNorm2d(a)

        self.conv2 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=4, padding=1, stride=2)
        self.norm2 = nn.BatchNorm2d(b)

        self.conv4 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=10)
        #self.norm4 = nn.BatchNorm2d(d)

        # self.conv5 = nn.Conv2d(in_channels=d, out_channels=e, kernel_size=3, padding=1, stride=2)
        # self.norm5 = nn.BatchNorm2d(e)

        # self.conv6 = nn.Conv2d(in_channels=e, out_channels=f, kernel_size=3, padding=1, stride=2)
        # self.norm6 = nn.BatchNorm2d(f)

        # self.conv7 = nn.Conv2d(in_channels=f, out_channels=g, kernel_size=4)

    def forward(self, x):

        # print(x.shape)
        x = self.relu(self.norm1(self.conv1(x)))#66
        # print(x.shape)
        x = self.relu(self.norm2(self.conv2(x)))#21
        # print(x.shape)
        #x = self.relu(self.norm3(self.conv3(x)))#6
        #print(x.shape)
        x = self.conv4(x)#1
        # print(x.shape)
        #x = self.relu(self.norm5(self.conv5(x)))#7
        #print(x.shape)
        #x = self.relu(self.norm6(self.conv6(x)))#4

        #x = self.relu(self.conv7(x))

        #print("end of encoder", x.shape)

        return self.sigmoid(x)
    
class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.deconv1 = nn.ConvTranspose2d(in_channels=g, out_channels=f, kernel_size=4)
        # self.norm1 = nn.BatchNorm2d(f)

        # self.deconv2 = nn.ConvTranspose2d(in_channels=f, out_channels=e, kernel_size=3, padding=1, stride=2)
        # self.norm2 = nn.BatchNorm2d(e)

        # self.deconv3 = nn.ConvTranspose2d(in_channels=e, out_channels=d, kernel_size=3, padding=1, stride=2)
        # self.norm3 = nn.BatchNorm2d(d)

        self.deconv4 = nn.ConvTranspose2d(in_channels=c, out_channels=b, kernel_size=10)
        self.norm4 = nn.BatchNorm2d(b)

        #self.deconv5 = nn.ConvTranspose2d(in_channels=c, out_channels=b, kernel_size=4, padding=1, stride=2)
        #self.norm5 = nn.BatchNorm2d(b)

        self.deconv6 = nn.ConvTranspose2d(in_channels=b, out_channels=a, kernel_size=4, padding=1, stride=2)
        self.norm6 = nn.BatchNorm2d(a)

        self.deconv7 = nn.ConvTranspose2d(in_channels=a, out_channels=3, kernel_size=4, padding=1, stride=2)

    def forward(self, x):

        #x = self.relu(self.norm1(self.deconv1(x)))
        #print(x.shape)
        #x = self.relu(self.norm2(self.deconv2(x))) 
        #print(x.shape)
        #x = self.relu(self.norm3(self.deconv3(x))) 
        #print(x.shape)
        x = self.relu(self.norm4(self.deconv4(x))) 
        #print(x.shape)
        #x = self.relu(self.norm5(self.deconv5(x)))

        x = self.relu(self.norm6(self.deconv6(x)))

        x = self.deconv7(x)
        #print("end of decoder", x.shape)

        return self.sigmoid(x)
    
class Encdec(pl.LightningModule):
    def __init__(self,):
        super().__init__()

        self.epochh=0

        self.encoder = Encoder()#.to(mps_device)
        self.decoder = Decoder()#.to(mps_device)


    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _step(self, batch, batch_idx):

        outputs = self(batch)
        #print("for loss:", outputs.shape, batch.shape)
        loss = nn.functional.mse_loss(outputs, batch)
        return loss
    

    def training_step(self, batch, batch_idx):

        # Called to compute and log the training loss
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=PARALLEL_TRAINING)
        return loss
    
    def validation_step(self, batch, batch_idx):

        # Called to compute and log the validation loss
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=PARALLEL_TRAINING)
        return val_loss

    def configure_optimizers(self):

        # Optimizer and LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    # def on_train_epoch_start(self):
    #     torch.save(self.encoder.state_dict(), "chkpt/bigger_encoder.chkpt")
    #     torch.save(self.decoder.state_dict(), "chkpt/bigger_decoder.chkpt")
        
    #     with torch.no_grad():

    #         image = test_dataset[0]#.to(mps_device)
    #         image = torch.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
            
    #         if not os.path.exists('predictions_bigger_encdec'):
    #             os.makedirs('predictions_bigger_encdec')

    #         print("Before epoch start")
    #         image=self.encoder(image)
    #         image=self.decoder(image)
    #         print("After on epoch start")

    #         pil_image = TF.to_pil_image(image[0,:,:,:])

    #         image_path = os.path.join('predictions_bigger_encdec', f"image_{self.epochh}.png")
    #         pil_image.save(image_path, inplace=True)
            

    #     self.epochh+=1
    #     self.train()

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

dir = os.getcwd() + '/dataset/CAT_00_treated'

image_files = os.listdir(dir)
random.shuffle(image_files)

train_files, test_files = train_test_split(image_files, test_size=0.3)  # Adjust test_size as needed
#print(len(train_files))
train_dataset = ImageDataset(dir, transform, file_list=train_files)
test_dataset = ImageDataset(dir, transform, file_list=test_files)

dataloader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=19, pin_memory=True, persistent_workers=True) # Added num_workers to avoid bottleneck
dataloader_test = DataLoader(test_dataset, batch_size=batches, shuffle=True, num_workers=19, pin_memory=True, persistent_workers=True)

print("before mps device")
#Device:
if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("Using Device: ", mps_device)

else:

        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                                                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                                                                  "and/or you do not have an MPS-enabled device on this machine.")

        if torch.cuda.is_available():
            mps_device = torch.device("cuda")
            print("Using device: ", mps_device)

        else:
            mps_device = torch.device("cpu")
            print("Using device: ", mps_device)


model = Encdec()
#model.to(mps_device)
torch.set_float32_matmul_precision('high') # Only if Tensor cores is used; Remember to set back to default after training

#print("model init done")

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")

if mps_device == torch.device("mps"):
    lightning_accelerator = "mps"
elif mps_device == torch.device("cuda"):
    lightning_accelerator = "gpu"
else:
    lightning_accelerator = "cpu"

if PARALLEL_TRAINING:
    lightning_accelerator = "gpu"
    number_nodes = 1#2
    number_gpus_per_node = 1
    strategy_ddp = DDPStrategy(gradient_as_bucket_view=True)
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=epochs, accelerator=lightning_accelerator, devices=number_gpus_per_node, num_nodes=number_nodes, strategy=strategy_ddp, logger=mlf_logger, log_every_n_steps=1, profiler="simple")
else:
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=epochs, accelerator=lightning_accelerator, logger=mlf_logger, log_every_n_steps=1, profiler='simple')
    
try:
    trainer.fit(model, dataloader_train, dataloader_test)
except:
    trainer.print(torch.cuda.memory_summary())
