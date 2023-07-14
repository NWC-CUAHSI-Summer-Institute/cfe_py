"""A file to store the function where we read the input data"""
import logging

from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import (
    TypeVar,
)
from datetime import datetime

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # TODO ADD THE FORCING (x) AND OBS (Y) VARS

        # Read in start and end datetime
        self.start_time = datetime.strptime(
            cfg["src\data"]["start_time"], r"%Y-%m-%d %H:%M:%S"
        )
        self.end_time = datetime.strptime(
            cfg["src\data"]["end_time"], r"%Y-%m-%d %H:%M:%S"
        )

        # Read forcing data into pandas dataframe
        forcing_df_ = pd.read_csv(cfg["src\data"]["forcing_file"])
        forcing_df_.set_index(pd.to_datetime(forcing_df_["date"]), inplace=True)
        self.forcing_df = forcing_df_[self.start_time : self.end_time].copy()

        # # Convert pandas dataframe to PyTorch tensors
        # Convert units
        # (precip/1000)   # kg/m2/h = mm/h -> m/h
        # (pet/1000/3600) # kg/m2/h = mm/h -> m/s

        """Numpy implementation
        precip = np.array([self.forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm])
        pet = np.array([self.forcing_df["potential_evaporation"].values / cfg.conversions.m_to_mm/ cfg.conversions.hr_to_sec])
        """

        precip = torch.tensor(
            self.forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm,
            device=cfg.device,
        )
        pet = torch.tensor(
            self.forcing_df["potential_evaporation"].values
            / cfg.conversions.m_to_mm
            / cfg.conversions.hr_to_sec,
            device=cfg.device,
        )

        """Numpy implementation
        x_ = np.stack([precip, pet])
        x_tr = x_.transpose()
        """

        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        x_tr = x_.transpose(0, 1)
        self.x = x_tr

        # Creating a time interval
        time_values = self.forcing_df["date"].values
        self.timestep_map = {time: idx for idx, time in enumerate(time_values)}

        # # TODO FIND OBSERVATION DATA TO TRAIN AGAINST
        obs_q_ = pd.read_csv(cfg["src\data"]["compare_results_file"])
        obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
        self.obs_q = obs_q_[self.start_time : self.end_time].copy()

        """Numpy implementation
        self.y = self.obs_q['QObs(mm/h)'].values
        """

        self.n_timesteps = len(self.obs_q)
        self.y = torch.tensor(self.obs_q["QObs(mm/h)"].values, device=cfg.device)

    def __getitem__(self, index) -> T_co:
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed data for a particular index
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """
        Method from the torch.Dataset parent class
        """
        return self.x.shape[0]
