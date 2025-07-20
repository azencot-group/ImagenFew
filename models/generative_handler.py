import torch
from abc import ABC, abstractmethod
from distributed.DDP import eDDP

class generativeHandler(ABC):
    def __init__(self, args, rank=None):
        self.args = args
        self.device = rank if (rank is not None) else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self.build_model().to(self.device)
        if self.args.ddp and (rank is not None):
            self.model = eDDP(self._model, device_ids=[self.device], output_device=self.device, find_unused_parameters=True)
        else:
            self.model = self._model

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_iter(self, train_dataloader):
        pass

    @abstractmethod
    def sample(self, n_samples, class_label, class_metadata):
        pass

    def save_model(self, ckpt_dir):
        torch.save(self._model.state_dict(), ckpt_dir)