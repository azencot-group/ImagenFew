from .kovae_cond import KoVAE_COND
from ..generative_handler import generativeHandler

import os
import torch
import logging

class KoVAEWrapper(KoVAE_COND):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x_ts, class_indices, num_classes):
        oh_class_labels = torch.nn.functional.one_hot(class_indices, num_classes=num_classes)
        x_rec, Z_enc, Z_enc_prior = super().forward(x_ts, oh_class_labels.to(torch.float32))
        loss = self.loss(x_ts, x_rec, Z_enc, Z_enc_prior)
        return loss[0], {'loss': loss[0].item()}
    
class Handler(generativeHandler):

    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def build_model(self):
        self.model = KoVAEWrapper(self.args)
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model
    
    def train_iter(self, train_dataloader, logger):
        for _, data in enumerate(train_dataloader, 1):
            x_ts = data[0].to(self.args.device).to(torch.float32)
            class_indices = data[1].to(self.args.device)
            self.optimizer.zero_grad()
            loss = self.model(x_ts, class_indices, self.args.n_classes)
            if len(loss) == 2:
                loss, to_log = loss
                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, self.epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

    def sample(self, n_samples, class_label, class_metadata):
        generated_set = []
        for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
            class_labels = torch.full((sample_size,), class_label, device=self.args.device)
            oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes)
            x_ts = self.model.sample_data(sample_size, oh_class_labels.to(torch.float32))
            generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)
    
    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. "
                            f"Returned the same state as input")
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            self.model.load_state_dict(loaded_state, strict=False)
            logging.warning("Successfully loaded previous state")