import torch
import torch.utils.data as Data
import functools


#NOTE this list holds all possible datasets
pre_train_index = 19
dataset_list = [
            "stock",
            "energy",
            "ETTh1",
            "Exchange",
            "MSL",
            "PSM",
            "SMAP",
            "SMD",
            "SelfRegulationSCP2",
            "UWaveGestureLibrary",
            "ECG5000",
            "NonInvasiveFetalECGThorax1",
            "Blink",
            "ElectricDevices",
            "Trace",
            "FordB",
            "EMOPain",
            "Chinatown",
            "SharePriceIncrease",
        # FINE TUNE DATASETS BELOW
            "sine",
            "mujoco",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Weather",
            "ILI",
            "SaugeenRiverFlow",
            "ECG200",
            "SelfRegulationSCP1",
            "StarLightCurves",
            "AirQuality",
            "Electricity",
        ]

def get_pretrained_datasets_names():
    return dataset_list[:pre_train_index]
class CombinedShortRangeDataset(Data.Dataset):
    def __init__(self, train_sets,test_sets,metadatas, args):
        # testsets are the datasets that the model will be tested on
        self.datasets = test_sets
        self.metadatas = metadatas
        self.max_channels = functools.reduce(lambda acc, metadata: max(acc, metadata['channels']), metadatas.values(), args.input_channels if args.input_channels is not None else 1)
        self.data = []
        self.class_labels = []
        self.num_datasets = len(dataset_list)
        for  dataset_name, dataset in train_sets.items():
            #NOTE: only trained_on datasets will be included in the multi-dataset
            current_channels = dataset.size(-1)
            if current_channels < self.max_channels:
                padding = torch.zeros(*dataset.size()[:-1], self.max_channels - current_channels, dtype=dataset.dtype, device=dataset.device)
                dataset = torch.cat((dataset, padding), dim=-1)
            self.data.append(dataset)
            self.class_labels.append(torch.full((len(dataset),), dataset_list.index(dataset_name), dtype=torch.long))
        self.data = torch.cat(self.data)
        self.class_labels = torch.cat(self.class_labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.class_labels[idx]
    def gen_dataloader(self, dataset_name, batch_size):
        """
        Args:
            args: arguments
        Returns:
            dataloader for the specified dataset along with its corresponding class index
        """
        dataset = Data.TensorDataset(self.datasets[dataset_name])
        return Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0), dataset_list.index(dataset_name)
