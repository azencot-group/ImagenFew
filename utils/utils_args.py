from omegaconf import OmegaConf
import argparse
import uuid

def parse_args_uncond():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--run_id', type=str, default=str(uuid.uuid4()), help='run id')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--model_ckpt', help='path to model checkpoint')
    parser.add_argument('--neptune', action='store_true', help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'unconditional'], help='tags for neptune logger', nargs='+')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--no_test_model', action='store_true', help='path to save checkpoints')

    # --- diffusion process --- #
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', help='deterministic sampling')

        # ------ KoVAE -------
    parser.add_argument('--pinv_solver', type=bool, default=False)

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/unconditional/stocks.yaml', help='config file')

    # --- handler ---
    parser.add_argument('--handler', type=str, help='Handler location')

    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--ddp', action='store_true', help='enable distributed training')
    parser.add_argument('--pretrain', action='store_true', help='Indicating pre-train setup')

    # --- data ---:
    parser.add_argument('--subset_p', type=float, help='fraction of the dataset to include')
    parser.add_argument('--subset_n', type=float, help='number of samples from dataset to include')
    parser.add_argument('--seq_len', type=int, help='input sequence length,')
    parser.add_argument('--train_on_datasets', help='datasets to train on',)

        # --- KoVAE ---
    parser.add_argument('--missing_value', type=float, default=0.)

    # --- image transformations ---:
    parser.add_argument('--use_stft', type=bool, help='use stft transform - if absent, use delay embedding')
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int, help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int, help='embedding for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--unconditional_rate', type=float, help='unconditional rate', default=0.1)

    # --- model--- :
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int, help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--channel_mult_emb', type=int, help='number of unet channels')
    parser.add_argument('--num_blocks', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')
    parser.add_argument('--lora_dim', type=int, default=4, help='lora dim')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--ft_method', type = str, default = 'all', help='ft method', choices=['all', 'lora', 'attn', 'non_attn','decoder'])
    parser.add_argument('--dynamic_size', type=int, default = [128,128], help='ch mut', nargs='+')

        # --- KoVAE ---
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--z_dim', type=int)
    parser.add_argument('--hidden_dim', type=int, help='the hidden dimension of the output decoder lstm')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, help='number of iterations between logging')
    parser.add_argument('--eval_metrics', type=str, default=["disc", "contextFID", "pred"], choices=["disc", "contextFID", "pred"], help='Evaluate metrics to compute', nargs='+')

        # -- KoVAE ---
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--w_rec', type=float, default=1.)
    parser.add_argument('--w_kl', type=float, default=.007)
    parser.add_argument('--w_pred_prior', type=float, default=0.005)



    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    parser.set_defaults(**config)
    return parser.parse_args()


def parse_args_cond():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'conditional'],
                        help='tags for neptune logger', nargs='+')

    # --- diffusion process ---
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/interpolation/TS2I/physionet.yaml',
                        help='config file')

    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---
    parser.add_argument('--dataset',
                        choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain',
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---
    parser.add_argument('--use_stft', type=bool,
                        help='use stft transform - if absent, use delay embedding')  # can be base
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')

    # --- model---
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')
    parser.add_argument('--dynamic_size', type=int, help='ch mut', nargs='+')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=100,
                        help='number of iterations between logging')

    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    return parsed_args
