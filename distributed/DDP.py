from torch.nn.parallel import DistributedDataParallel as DDP

class eDDP(DDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        return super().forward(*inputs, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)