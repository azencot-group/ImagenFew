from .ImagenTime import ImagenTime
from .sampler import DiffusionProcess
from ..generative_handler import generativeHandler

import os
import torch
import logging
import torch.nn as nn

class Handler(generativeHandler):

    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def build_model(self):
        self.model = ImagenTime(self.args, self.args.device)
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model
    
    def train_iter(self, train_dataloader, logger):
        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()

            # Time series & mask
            x_ts = data[0].to(self.args.device)

            # Convert time series & mask to image
            x_img = self.model.ts_to_img(x_ts)
            class_indices = data[1].to(self.args.device)

            labels = nn.functional.one_hot(class_indices, num_classes = self.args.n_classes)
            output, weight = self.model(x_img, labels = labels)
            time_loss   = (output - x_img).square()
            loss = (weight * (time_loss)).mean()
            logger.log(f'train/karras loss', loss.detach().item(), self.epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            self.model.on_train_batch_end()

    def sample(self, n_samples, class_label, class_metadata):
        generated_set = []
        with self._model.ema_scope():
            self.process = DiffusionProcess(self.args, self._model.net, (self.args.input_channels, self.args.img_resolution, self.args.img_resolution))
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                class_labels = torch.full((sample_size,), class_label, device=self.args.device)
                oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes)
                x_img_sampled = self.process.sampling(sample_size, class_labels = oh_class_labels)
                x_ts = self._model.img_to_ts(x_img_sampled)[:,:,:class_metadata['channels']]
                generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)
    
    def save_model(self, ckpt_dir):
        state = {'model': self._model.state_dict()}
        if self.args.ema is not None:
            state['ema_model'] = self._model.model_ema.state_dict()
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. "
                            f"Returned the same state as input")
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            self.model.load_state_dict(loaded_state['model'], strict=True)
            if 'ema_model' in loaded_state and self.args.ema is not None:
                self.model.model_ema.load_state_dict(loaded_state['ema_model'], strict=True)
            logging.info(f'Successfully loaded previous state')