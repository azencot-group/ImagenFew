import torch
import torch.nn as nn
from contextlib import contextmanager
from .networks import EDMPrecond
from .ema import LitEma
from .img_transformations import STFTEmbedder, DelayEmbedder
import loralib as lora

class ImagenTime(nn.Module):
    def __init__(self, args, device):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.num_classes = args.n_classes
        if not args.ft_method == 'lora':
            args.lora_dim = None
        self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution,
                              label_dim=self.num_classes, lora_rank = args.lora_dim)


        # delay embedding is used
        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len

            # NOTE: added this
            self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding)
        else:
            self.ts_img = STFTEmbedder(self.device, args.seq_len, args.n_fft, args.hop_length)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        """
        Args:
            signal: signal to convert to image
            pad_val: value to pad the image with, if delay embedding is used. Do not use for STFT embedding

        """
        # pad_val is used only for delay embedding, as the value to pad the image with
        # when creating the mask, we need to use 1 as padding value
        # if pad_val is given, it is used to overwrite the default value of 0
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    # init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
    def init_stft_embedder(self, train_loader):
        """
        Args:
            train_loader: training data

        caches min and max values for the real and imaginary parts
        of the STFT transformation, which will be used for normalization.
        """
        assert type(self.ts_img) == STFTEmbedder, "You must use the STFTEmbedder to initialize the min and max values"
        data = []
        for i, data_batch in enumerate(train_loader):
            data.append(data_batch[0])
        self.ts_img.cache_min_max_params(torch.cat(data, dim=0))

    def loss_fn(self, x, class_indicies, num_classes):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}


        labels = nn.functional.one_hot(class_indicies,num_classes = num_classes)
        output, weight = self.forward(x, labels = labels)

        time_loss   = (output - x).square()
        loss = (weight * (time_loss)).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def loss_fn_impute(self, x, mask, class_indicies, num_classes):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        labels = nn.functional.one_hot(class_indicies,num_classes = num_classes)
        output, time_weight = self.forward_impute(x, mask, labels = labels)

         # denoising matching term
        time_loss   = (output - x).square()
        loss = ((time_weight * time_loss) * (1 - mask)).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log

    def forward(self, x, labels=None, augment_pipe=None):

        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        D_yn = self.net(y + n, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_impute(self, x, mask, labels=None, augment_pipe=None):

        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # noisy impute part
        n = torch.randn_like(x) * sigma
        noise_impute = n * (1 - mask)
        x_to_impute = x * (1 - mask) + noise_impute

        # clear image
        x = x * mask
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

        D_yn = self.net(y + x_to_impute, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_forecast(self, past, future, labels=None, augment_pipe=None):
        s, e = past.shape[-1], future.shape[-1]
        rnd_normal = torch.randn([past.shape[0], 1, 1, 1], device=past.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(past) if augment_pipe is not None else (past, None)
        n = torch.randn_like(future) * sigma
        full_seq = self.pad_f(torch.cat([past, future + n], dim=-1))
        D_yn = self.net(full_seq, sigma, labels, augment_labels=augment_labels)[..., s:(s + e)]
        return D_yn, weight

    def pad_f(self, x):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (
            0, max_side - rows, 0, 0)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)

    def reset_ema_counter(self):
        self.model_ema.num_updates.zero_()

    def freeze_all_params(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze_embedding_matrix(self):
        for param in self.net.model.map_label.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        for name, param in self.net.named_parameters():
            if 'enc' in name:
                param.requires_grad = False
    def freeze_attn(self):
        for name, param in self.net.named_parameters():
            if 'proj' in name or 'qkv' in name:
                param.requires_grad = False

    def freeze_non_attn(self):
        for name, param in self.net.named_parameters():
            if 'proj' not in name and 'qkv' not in name and 'bias' not in name:
                param.requires_grad = False


    def freeze_decoder(self):
        for name, param in self.net.namesd_parameters():
            if 'dec' in name:
                param.requires_grad = False
    def setup_finetune(self, args):
        match args.ft_method:
            case 'lora':
                lora.mark_only_lora_as_trainable(self.net, bias = 'all')
                self.unfreeze_embedding_matrix()
                # self.model_ema.update_ema_params(self.net)
            case 'decoder':
                # fine-tune only the decoder
                self.freeze_encoder()
            case 'attn':
                # fine-tune only the attention layers
                self.freeze_attn()
            case 'non_attn':
                self.freeze_non_attn()
            case 'encoder':
                self.freeze_decoder()
            case 'all':
                pass
        self.model_ema.setup_finetune(self.net)


