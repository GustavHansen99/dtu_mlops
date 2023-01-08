from torch import nn, optim
from pytorch_lightning import LightningModule
import wandb

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3), # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3), # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3), # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3), # [N, 8, 20]
            nn.LeakyReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
        
        self.criterium = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        preds = preds.detach().numpy()
        self.logger.experiment.log({'logits': wandb.Histogram(preds)})
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 1e-2)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        val_loss = self.criterium(preds, target)
        val_acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        return val_loss

    def test_step(self, batch, batch_idx):
        return None