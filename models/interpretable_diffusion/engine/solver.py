import os
import sys
import time
from contextlib import contextmanager

import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
# from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from utils.io_utils import instantiate_from_config, get_model_parameters_info

from metrics import evaluate_model_uncond
from ..ema import LitEma
from utils.utils import save_checkpoint, restore_state

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, train_loader, test_loader, logger=None, multidata_dataset=None, dataloader=None, unpad_f=None):
        super().__init__()
        self.multidata_dataset = multidata_dataset
        self.model = model
        self.device = self.model.betas.device
        args.device = self.device
        self.epochs = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        # self.dl = cycle(dataloader['dataloader'])
        # self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataloader = dataloader
        self.unpad_f = unpad_f

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = LitEma(self.model, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        # self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        # if self.logger is not None:
        #     self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model)
        if context is not None:
            print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            self.ema.restore(self.model.parameters())
            if context is not None:
                print(f"{context}: Restored training weights")
    def train(self):
        state = dict(model=self.model, epoch=0)
        best_score = float('inf')
        device = self.device
        step = 0

        for epoch in range(step, self.epochs):
            for i, data in enumerate(self.train_loader, 1):
                # data = next(self.train_loader).to(device)
                x = data[0].to(device).float()

                class_indices = data[1].to(device)
                loss = self.model(x, class_indices, self.args.n_classes, target=x)
                    # loss = loss / self.gradient_accumulate_every
                loss.backward()

                self.logger.log('train/loss', loss.item(), self.step)

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(loss.item())
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema(self.model) #
            if self.logger is not None and epoch % self.log_frequency == 0:
                self.model.eval()
                scores_mean = []
                for dataset in self.args.trained_on_datasets:
                    # NOTE: changed to this
                    self.args.dataset = dataset
                    test_loader, class_label = self.multidata_dataset.gen_dataloader(dataset, self.args.test_batch_size)
                    gen_sig = []
                    real_sig = []
                    with torch.no_grad():
                        with self.ema_scope():
                            for data in tqdm(test_loader, desc=f"Evaluating {dataset}"):
                                class_labels = torch.full((data[0].shape[0],), class_label, device=device)
                                oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes).float()
                                sampled = self.sample(num = data[0].shape[0], shape=[data[0].shape[1], data[0].shape[2]], labels= oh_class_labels)
                                x_ts = self.unpad_f(sampled, dataset)
                                unpad_data = self.unpad_f(data[0], dataset)
                                gen_sig.append(x_ts.detach().cpu().numpy())
                                real_sig.append(unpad_data.detach().cpu().numpy())
                    gen_sig = np.vstack(gen_sig)
                    real_sig = np.vstack(real_sig)
                    scores = evaluate_model_uncond(real_sig, gen_sig, self.args)
                    scores_mean.append(scores[f'{dataset}_disc_mean'])
                    for key, value in scores.items():
                        self.logger.log(f'test/{key}', value, epoch)
                self.logger.log(f'test/disc_mean', np.mean(scores_mean), epoch)
                # --- save checkpoint ---
                if np.mean(scores_mean) < best_score:
                    best_score = np.mean(scores_mean)
                    ema_model = self.ema
                    save_checkpoint(self.args.log_dir, state, epoch , ema_model)


        print('training complete')
        # if self.logger is not None:
        #     self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))
    def finetune(self):
        state = dict(model=self.model, epoch=0)
        ema_model = self.ema
        restore_state(self.args.model_ckpt, state, ema_model=self.ema)
        ema_model.setup_finetune(self.model)
        self.train()

    def sample(self, num , shape=None, model_kwargs=None, cond_fn=None, labels = None):
        if self.logger is not None:
            tic = time.time()
            # self.logger.info('Begin to sample...')
        # samples = np.empty([0, shape[0], shape[1]])
        # num_cycle = int(num // size_every) + 1

        # for _ in range(num_cycle):
            # sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
        samples = self.model.generate_mts(batch_size=num, model_kwargs=model_kwargs, cond_fn=cond_fn, labels = labels)
        # samples = np.row_stack([samples, sample.detach().cpu().numpy()])
        torch.cuda.empty_cache()

        # if self.logger is not None:
            # self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

