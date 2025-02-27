import argparse
import sys

import torch

from data import CorruptMnist
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import matplotlib.pyplot as plt
import wandb

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        #wandb.init(project='mlops_exercises4')
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=1e-3)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()  # this is our LightningModule
        model = model.to(self.device)
        train_set = CorruptMnist(train=True)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
        val_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)

        checkpoint_callback = ModelCheckpoint(
            dirpath="src/models/checkpoints/", monitor="val_loss", mode="min"
            )
        trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=5, logger=pl_loggers.WandbLogger(project="mlops_exercises4"))
        trainer.fit(model, train_dataloader, val_dataloader)
        #trainer.validate(model, val_dataloader)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        

        n_epoch = 5
        columns=["index", "image", "guess", "truth"]
        test_dt = wandb.Table(columns = columns)
        for epoch in range(n_epoch):
            loss_tracker = []
            counter = 0
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                preds = model(x.to(self.device))
                loss = criterion(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                wandb.log({"loss": loss})
                loss_tracker.append(loss.item())
                # log training progress in terms of preds and images
                if counter % 50 == 0:
                    preds = preds.argmax(dim=-1)
                    for idx, image in enumerate(x):
                        row = [idx, wandb.Image(image), preds[idx], y[idx]]
                        test_dt.add_data(*row)
                counter += 1
            print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
        wandb.log({"mnist_predictions": test_dt})
        torch.save(model.state_dict(), 'src/models/trained_model.pt')
            
        plt.plot(loss_tracker, '-')
        plt.xlabel('Training step')
        plt.ylabel('Training loss')
        plt.savefig("src/visualization/training_curve.png")
        """
        
        return None
            
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model = model.to(self.device)

        test_set = CorruptMnist(train=False)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(self.device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(self.device)).sum().item()
            total += y.numel()
            
        print(f"Test set accuracy {correct/total}")


if __name__ == '__main__':
    TrainOREvaluate()
