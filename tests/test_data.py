from tests import _PATH_DATA
from src.models.data import CorruptMnist
import torch
import numpy as np

def test_data_samples():
    train = CorruptMnist(train=True)
    test = CorruptMnist(train=False)

    assert len(train) == 25000 and len(test) == 5000, "The dataset does not contain the correct number of samples"
    assert (
        all([train_element.shape == torch.Size([1, 28, 28]) for train_element, _ in train]) 
        and 
        all([test_element.shape == torch.Size([1, 28, 28]) for test_element, _ in test])), "The shape of the data samples does not match the required"

def test_data_targets():
    train = CorruptMnist(train=True)
    test = CorruptMnist(train=False)
    assert (
        len(np.unique(train.targets)) == 10 
        and 
        len(np.unique(test.targets)) == 10), "Not all classes are represented in the data"