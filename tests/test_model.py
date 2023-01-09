from src.models.model import MyAwesomeModel
import torch

batch_size = 128
model = MyAwesomeModel()
sample = torch.rand(batch_size, 1, 28, 28)
output = model.forward(sample)
assert output.shape == torch.Size([batch_size, 10])