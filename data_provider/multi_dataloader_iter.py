import numpy as np
import torch
from .combined_datasets import dataset_list


class MultiDataloaderIter:
    def __init__(self, trainloaders, testsets):
        self.trainloaders = trainloaders
        self.testsets = testsets
        self._lens = [len(dataloader) for _, (dataloader, _) in self.trainloaders.items()]
        self.num_datasets = len(dataset_list)

    def __iter__(self):
        self.active_iters = [(iter(dataloader), metadata) for key, (dataloader, metadata) in self.trainloaders.items()]
        self.iteration = 0
        return self

    def __next__(self):
        sample, metadata = self.__get_next()
        self.iteration += 1
        return sample, torch.full((sample.size(0),), fill_value=dataset_list.index(metadata['name']))
    
    def __get_next(self):
        index = np.random.randint(0, len(self.active_iters))
        try:
            return next(self.active_iters[index][0]), self.active_iters[index][1]
        except StopIteration:
            del self.active_iters[index]
            if len(self.active_iters) == 0:
                raise StopIteration
            index = np.random.randint(0, len(self.active_iters))
            return self.__get_next()
        
    def __len__(self):
        return sum(self._lens)

    def gen_dataloader(self, dataset_name):
        return self.testsets[dataset_name], dataset_list.index(dataset_name)
    