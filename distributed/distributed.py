from torch.distributed import (init_process_group, destroy_process_group)
from datetime import timedelta


import torch.multiprocessing as mp
import torch.distributed as dist
import random
import socket
import torch
import os

class Disributed():
    def __init__(self, train_fn) -> None:
        self.train_fn = train_fn

    def run(self, args):
        if args.ddp:
            args.world_size = torch.cuda.device_count()
            args.gpu_num = args.world_size
        else:
            args.world_size = 1
            args.gpu_num = 1
        port = str(self.next_free_port())
        print(port)
        mp.spawn(self.train_rank, args=(args, port), nprocs=args.world_size, join=True)

    def train_rank(self, rank, args, port):
        args.device = f"cuda:{rank}"
        self.ddp_setup(rank, args.gpu_num, port)
        self.train_fn(args)
        destroy_process_group()

    def ddp_setup(self, rank, world_size, port="12355"):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        print(f"Start running basic DDP on rank {rank}.")
        dist.barrier()

    def next_free_port(self, port=12355, max_port=65535):
        rand_offset = random.randint(0, 1000)
        port += rand_offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while port <= max_port:
            try:
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                port += 1
        raise IOError('no free ports')
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0