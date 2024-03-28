import lightning as L
import torch

from lightning.pytorch.demos import Transformer
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pathlib

PARALLEL_TRAINING = False
epochs = 10

class LightningTransformer(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

dataset = WikiText2(pathlib.Path('/Data/temp/'))
dataloader = DataLoader(dataset)
model = LightningTransformer(vocab_size=dataset.vocab_size)

if PARALLEL_TRAINING:
    lightning_accelerator = "gpu"
    number_nodes = 2
    number_gpus_per_node = 1
    strategy_ddp = DDPStrategy(gradient_as_bucket_view=True)
    trainer = L.Trainer(max_epochs=epochs, accelerator=lightning_accelerator, devices=number_gpus_per_node, num_nodes=number_nodes, strategy=strategy_ddp, profiler="simple")
else:
    trainer = L.Trainer(max_epochs=epochs, accelerator="gpu", profiler="simple")
trainer.fit(model=model, train_dataloaders=dataloader)