import torch
import wget
from torch.utils.data import Dataset
import numpy as np
import os

class CorruptMnist(Dataset):
    def __init__(self, train):
        self.raw_path = "src/data/raw/"
        self.processed_path = "src/data/processed/"
        #dvc_files = [dvc_file for dvc_file in os.listdir("data/") if dvc_file.endswith(".dvc")]
        #if len(dvc_files) == 0:
        npz_files = [npz_file for npz_file in os.listdir(self.raw_path) if npz_file.endswith(".npz")]
        if len(npz_files) == 0:
            self.download_data(train)
        #data = dvc.api.read("data/corruptmnist.dvc", mode='rb')
        #data = pickle.loads(data)
        if train:
            content = [ ]
            for i in range(5):
                content.append(np.load(os.path.join(self.raw_path, f"train_{i}.npz"), allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(os.path.join(self.raw_path, f"test.npz"), allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def download_data(self, train):
        if train:
            for file_idx in range(5):
                wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz", 
                            os.path.join(self.raw_path, f"train_{file_idx}.npz"))
        else:  
            wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz", 
                        os.path.join(self.raw_path, "test.npz"))
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)