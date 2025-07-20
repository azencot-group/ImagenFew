from pathlib import Path
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.repository import get_dataset
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def GLUONTS(**config):
    return GLUONTSDataset(
        dataset_name=config['dataset_name'],
        size=(config['seq_len'], config['label_len'], config['pred_len']),
        path=os.path.join(config['datasets_dir'], config['rel_path']),
        features=config["features"],
        flag=config["flag"],
    )

default_dataset_writer = JsonLinesWriter()


class GLUONTSDataset(Dataset):
    """
    NOTE: added flags for splits, multivariate timeseries, and normalization

    Copied from GLUONTS:

    Get a repository dataset.

    The datasets that can be obtained through this function have been used
    with different processing over time by several papers (e.g., [SFG17]_,
    [LCY+18]_, and [YRD15]_) or are obtained through the `Monash Time Series
    Forecasting Repository <https://forecastingdata.org/>`_.

    Parameters
    ----------
    dataset_name
        Name of the dataset, for instance "m4_hourly".
    regenerate
        Whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        Where the dataset should be saved.
    prediction_length
        The prediction length to be used for the dataset. If None, the default
        prediction length will be used. If the dataset is already materialized,
        setting this option to a different value does not have an effect.
        Make sure to set `regenerate=True` in this case. Note that some
        datasets from the Monash Time Series Forecasting Repository do not
        actually have a default prediction length -- the default then depends
        on the frequency of the data:
        - Minutely data --> prediction length of 60 (one hour)
        - Hourly data --> prediction length of 48 (two days)
        - Daily data --> prediction length of 30 (one month)
        - Weekly data --> prediction length of 8 (two months)
        - Monthly data --> prediction length of 12 (one year)
        - Yearly data --> prediction length of 4 (four years)

    Returns
    -------
        Dataset obtained by either downloading or reloading from local file.
    """

    default_pred_lens = {
        "exchange_rate": 30,
        "solar-energy": 24,
        "electricity": 24,
        "traffic": 24,
        "exchange_rate_nips": 30,
        "electricity_nips": 24,
        "traffic_nips": 24,
        "solar_nips": 24,
        "wiki2000_nips": 30,
        "wiki-rolling_nips": 30,
        "taxi_30min": 24,
        "kaggle_web_traffic_without_missing": 59,
        "kaggle_web_traffic_weekly": 8,
        "m1_yearly": 10,
        "m1_quarterly": 8,
        "m1_monthly": 18,
        "nn5_daily_without_missing": 56,
        "nn5_weekly": 8,
        "tourism_monthly": 24,
        "tourism_quarterly": 8,
        "tourism_yearly": 4,
        "cif_2016": 12,
        "wind_farms_without_missing": 60,
        "car_parts_without_missing": 12,
        "dominick": 8,
        "fred_md": 12,
        "pedestrian_counts": 48,
        "hospital": 12,
        "covid_deaths": 30,
        "kdd_cup_2018_without_missing": 48,
        "weather": 30,
        "m3_monthly": 18,
        "m3_quarterly": 8,
        "m3_yearly": 6,
        "m3_other": 8,
        "m4_hourly": 48,
        "m4_daily": 14,
        "m4_weekly": 13,
        "m4_monthly": 18,
        "m4_quarterly": 8,
        "m4_yearly": 6,
        "uber_tlc_daily": 7,
        "uber_tlc_hourly": 24,
        "airpassengers": 12,
        "australian_electricity_demand": 60,
        "electricity_hourly": 48,
        "electricity_weekly": 8,
        "rideshare_without_missing": 48,
        "saugeenday": 30,
        "solar_10_minutes": 60,
        "solar_weekly": 5,
        "sunspot_without_missing": 30,
        "temperature_rain_without_missing": 30,
        "vehicle_trips_without_missing": 30,
    }

    def __init__(self,
                 dataset_name: str,
                 size: tuple,
                 path="../dataset/gluonts",
                 dataset_writer=default_dataset_writer,
                 features="S",
                 flag="train",
                 scale=True,
                 ):

        path = Path(path)

        assert dataset_name in self.default_pred_lens.keys(
        ), "{} dataset not recognized".format(dataset_name)

        if size is None:  # Hardcoded behavior, we can change via setting size
            self.seq_len = default_pred_lens[dataset_name] * 2
            self.label_len = 0
            self.pred_len = default_pred_lens[dataset_name]
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        try:  # Tries first to not regenerate, but does it if needed
            self.gluonts_dataset = get_dataset(
                dataset_name=dataset_name,
                path=path,
                regenerate=False,
                dataset_writer=dataset_writer
            )
        except:
            print('Regenerating {}...'.format(dataset_name))
            self.gluonts_dataset = get_dataset(
                dataset_name=dataset_name,
                path=path,
                regenerate=True,
                dataset_writer=dataset_writer
            )

        self.scale = scale
        self.features = features  # "S" - singlevariate or "M" - mulitvariate

        # Will need to do splitting internally or externally after the below steps

        # Getting test gives you all the data. .train is just downsampled version
        x = []
        times = []
        for inp_dict in self.gluonts_dataset.test:
            x.append(inp_dict['target'])
            times.append(inp_dict['start'])
            # May need to look into quicker method if the for loop turns out to be slow

        if self.features == "M":
            # multivariate, dataset becomes just one series, where each series is a new variable
            x = np.stack(x, axis=1)

            # start and end borders for train, val, and test splits
            # start_borders = [0, int(x.shape[0] * 0.8 - self.seq_len), int(x.shape[0] * 0.9 - self.seq_len)]
            # end_borders = [int(x.shape[0] * 0.8), int(x.shape[0] * 0.9), x.shape[0]]
            start_borders = [
                0, int(x.shape[0] * 0.8 - self.seq_len), int(x.shape[0] * 0.8 - self.seq_len)]
            end_borders = [int(x.shape[0] * 0.8), x.shape[0], x.shape[0]]
            set_type = {"train": 0, "val": 1, "test": 2}[flag]
            start = start_borders[set_type]
            end = end_borders[set_type]

            if scale:
                self.scaler = StandardScaler()
                train_start, train_end = start_borders[0], end_borders[0]
                x_train = x[train_start:train_end]
                self.scaler.fit(x_train)
                x = self.scaler.transform(x)

            self.data_x = x[start:end]
            self.data_x = torch.from_numpy(self.data_x)

        elif self.features == "S":
            raise NotImplementedError(
                "need to implement train/test/val split and scaling for single variable")
            self.Xtmp = x
            x_pad, pad_mask = pad_and_stack(x)
            self.data_x = context_based_split(
                x_pad, pad_mask, context_len=self.seq_len + self.pred_len)
            self.data_x = torch.from_numpy(self.data_x)
            if len(self.data_x.shape) == 2:
                # Make multivariate with one sensor
                self.data_x = self.data_x.unsqueeze(-1)

        self.scaler = StandardScaler()

        self.times = times  # Need to transform ------ Not used for now

    def __getitem__(self, ind):
        if self.features == "S":
            # Another thing: there are multple samples, thus we can't just index from data_x like below and in other datasets
            s_begin = 0  # Always 0 bc we collapse down into individual samples
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[ind, s_begin:s_end, :]
            # y is drawn from x - they're the same sequence
            seq_y = self.data_x[ind, r_begin:r_end, :]
            seq_x_mark = torch.zeros_like(seq_x)
            seq_y_mark = torch.zeros_like(seq_y)

        elif self.features == "M":
            s_begin = ind
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_x[r_begin:r_end]
            seq_x_mark = torch.zeros_like(seq_x)
            seq_y_mark = torch.zeros_like(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.features == "S":
            return self.data_x.shape[0]
        elif self.features == "M":
            return len(self.data_x) - self.seq_len - self.pred_len + 1


def pad_and_stack(arrays):
    """
    Pads and stacks a list of numpy arrays of varying lengths and creates a boolean array 
    indicating padded elements.

    Args:
    arrays (list of np.array): List of one-dimensional numpy arrays.

    Returns:
    np.array: A two-dimensional numpy array where each original array is padded with zeros
              to match the length of the longest array in the list.
    np.array: A two-dimensional boolean array where True indicates a padded element and 
              False indicates an original element.
    """
    # Find the maximum length among all arrays
    max_len = max(len(a) for a in arrays)

    # Initialize lists to hold padded arrays and boolean arrays
    padded_arrays = []
    boolean_arrays = []

    for a in arrays:
        # Amount of padding needed
        padding = max_len - len(a)

        # Pad the array and add it to the list
        padded_arrays.append(np.pad(a, (0, padding), mode='constant'))

        # Create a boolean array (False for original elements, True for padding)
        boolean_array = np.array([False] * len(a) + [True] * padding)
        boolean_arrays.append(boolean_array)

    # Stack the padded arrays and boolean arrays vertically
    stacked_array = np.vstack(padded_arrays)
    boolean_stacked = np.vstack(boolean_arrays)

    return stacked_array, boolean_stacked


def context_based_split(X, is_pad, context_len: int):
    split_inds = np.arange(start=0, stop=X.shape[1], step=context_len)

    X_collapse = []
    is_pad_collapse = []

    for i in range(1, len(split_inds)):
        X_collapse.append(X[:, split_inds[i-1]:split_inds[i]])
        is_pad_collapse.append(is_pad[:, split_inds[i-1]:split_inds[i]])

    Xnew = np.concatenate(X_collapse, axis=0)
    pad_new = np.concatenate(is_pad_collapse, axis=0)

    pad_by_sample = np.any(pad_new, axis=1)

    return Xnew[~pad_by_sample, :]