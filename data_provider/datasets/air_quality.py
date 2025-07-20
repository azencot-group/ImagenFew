from utils.utils_data import real_data_loading

import numpy as np
import torch

def AirQuality(seq_len, datasets_dir, **kwargs):
    ori_data = real_data_loading('AirQuality', seq_len, datasets_dir)
    return torch.Tensor(np.array(ori_data))