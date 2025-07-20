from utils.utils_data import sine_data_generation

import numpy as np
import torch

def Sine(seq_len, input_channels, **kwargs):
    ori_data = sine_data_generation(10000, seq_len, input_channels)
    return torch.Tensor(np.array(ori_data))