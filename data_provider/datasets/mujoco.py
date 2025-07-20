from utils.utils_data import MujocoDataset

import numpy as np
import torch
import os

def Mujoco(seq_len, path, datasets_dir, **kwargs):
    return torch.Tensor(MujocoDataset(seq_len, 'mujoco', os.path.join(datasets_dir, path), 0.0).original_sample)
