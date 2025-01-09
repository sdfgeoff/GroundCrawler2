
from typing import cast
from ai.vision.DynamicMemory import DynamicMemory, DataSet
import torch

def tensor_to_float_tensor(t: torch.Tensor) -> torch.FloatTensor:
    return cast(torch.FloatTensor, t.float())

def test_dataset_stores_items():
    d = DataSet([
        tensor_to_float_tensor(torch.zeros(3, 64, 64))
    ])
    assert d.images[0].shape == (3, 64, 64)
    assert len(d) == 1

