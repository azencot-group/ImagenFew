import numpy as np
import os
import logging
import torch
import collections

def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb

def log_config_and_tags(args, logger, name, multidata_dataset=False):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    if multidata_dataset:
        logger.add_tags(['multidata'])
    else:
        logger.add_tags([args.train_on_datasets[0]])


def get_model_name_diffusion():
    return f'diffusionTS'
def create_model_name_and_dir(args, extra_name=""):
    name = get_model_name(args)
    args.ts2vec_dir = os.path.join(args.log_dir ,"TS2VEC")
    os.makedirs(args.ts2vec_dir , exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, extra_name, os.path.splitext(os.path.basename(args.config))[0], args.run_id, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name

def get_model_name(args):
    name = f'{args.handler}'
    return name

def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)


# --- extrapolation and interpolation --- #
# get the mask and x for the time series
def get_x_and_mask(args, data):
    if args.dataset in ['climate', 'physionet']:
        # in the case of these datasets, the 'data_to_predict' is the same as 'observed_data
        if args.task == 'extrapolation':
            # concat the observed and predicted data
            x_ts = torch.cat([data['observed_data'], data['data_to_predict']], dim=1).to(args.device)
            # the predicted mask is opposite. the 1s are observed in the mask so it needed to be flipped in our case
            mask_ts = torch.cat([data['observed_mask'],  1 - data['mask_predicted_data']], dim=1).to(args.device)
        else:
            x_ts = data['observed_data'].to(args.device)
            mask_ts = data['mask_predicted_data'].to(args.device)
    else:
        if args.task == 'extrapolation':
            x_ts = data[0].float().to(args.device)
            # half ones and half zeros
            mask_ts = torch.zeros_like(x_ts)
            mask_ts[:, :x_ts.shape[1] // 2] = 1
        else:
            x_ts = data[0].float().to(args.device)
            # --- generate random mask and mask x as it time series --- #
            B, T, N = x_ts.shape
            mask_ts = torch.rand((B, T, N)).to(args.device)
            mask_ts[mask_ts <= args.mask_rate] = 0  # masked
            mask_ts[mask_ts > args.mask_rate] = 1  # remained

    return mask_ts, x_ts


def log_iter(epoch, loggin_iter):
    if isinstance(loggin_iter, collections.abc.Iterable):
        return epoch in loggin_iter
    elif isinstance(loggin_iter, int):
        return epoch % loggin_iter == 0
    else:
        return False