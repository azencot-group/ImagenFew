from utils.utils_data import real_data_loading

import numpy as np
import torch

def Stock(seq_len, datasets_dir, **kwargs):
    ori_data = real_data_loading('stock', seq_len, datasets_dir)
    return torch.Tensor(np.array(ori_data))
