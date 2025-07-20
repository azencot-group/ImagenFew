import os
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from omegaconf import OmegaConf
from models.generative_handler import generativeHandler
from models.interpretable_diffusion.ema import LitEma
from utils.io_utils import instantiate_from_config


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        self.config = OmegaConf.to_object(OmegaConf.load(args.config))
        super().__init__(args, rank)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['solver'].get('base_lr', 1e-4),
            betas=(0.9, 0.96),
        )
        self.ema = LitEma(self.model, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)

        self.scheduler = instantiate_from_config({
            **self.config['solver']['scheduler'],
            'params': {'optimizer': self.optimizer}
        })

        self.step = 0

    def build_model(self):
        return instantiate_from_config(self.config['model'])

    def train_iter(self, train_dataloader,logger):
        for _, data in enumerate(train_dataloader, 1):
            x = data[0].to(self.device).float()

            self.optimizer.zero_grad()
            loss = self.model(x, target=x)

            if isinstance(loss, tuple):
                loss, to_log = loss
                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, self.step)
            else:
                logger.log(f'train/loss', loss.item(), self.step)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(loss.item())
            self.ema(self.model)
            self.step += 1

    def sample(self, n_samples, class_label=None, class_metadata=None):
        with torch.no_grad():
            with self.ema_scope():
                return self.model.generate_mts(batch_size=n_samples)

    def ema_scope(self):
        class EMAScope:
            def __init__(self, model, ema):
                self.model = model
                self.ema = ema

            def __enter__(self):
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model)

            def __exit__(self, exc_type, exc_value, traceback):
                self.ema.restore(self.model.parameters())

        return EMAScope(self.model, self.ema)

    def save_model(self, ckpt_dir):
        torch.save(self._model.state_dict(), os.path.join(ckpt_dir, 'diffusionTS_model.pt'))