"""A file to store the function where we read the input data"""
import logging
from tqdm import tqdm
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

        # Read in start and end datetime
        self.start_time = datetime.strptime(
            cfg.data["start_time"], r"%Y-%m-%d %H:%M:%S"
        )

        self.end_time = datetime.strptime(cfg.data["end_time"], r"%Y-%m-%d %H:%M:%S")

        self.x = self.get_forcings(cfg)

        self.basin_attributes = self.get_attributes(cfg)

        if (cfg.run_type == "ML") | (cfg.run_type == "generate_synthetic"):
            self.y = self.get_observations(cfg)
        elif cfg.run_type == "ML_synthetic_test":
            self.y = self.get_synthetic(cfg)

        self.cfe_params = self.get_cfe_params(cfg)

    def __getitem__(self, index):
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed data for a particular timestep
        """
        return self.x[..., index, ...], self.y[..., index, ...]

    def __len__(self):
        """
        Method from the torch.Dataset parent class. Returns the number of timesteps
        """
        return self.x.shape[1]

    def get_forcings(self, cfg: DictConfig):
        basin_ids = cfg.data.basin_id
        # Calculate the time difference between end_time and start_time
        time_difference = self.end_time - self.start_time

        # Calculate the total number of hours between the two datetimes (adding 1 for index error)
        hours_difference = int(time_difference.total_seconds() / 3600) + 1
        output_tensor = torch.zeros([len(basin_ids), hours_difference, 2])
        # Read forcing data into pandas dataframe

        for i in tqdm(range(len(basin_ids)), desc="Reading forcings"):
            id = basin_ids[i]
            forcing_df_ = pd.read_csv(cfg.data.forcing_file.format(id))
            forcing_df_.set_index(pd.to_datetime(forcing_df_["date"]), inplace=True)
            forcing_df = forcing_df_[self.start_time : self.end_time].copy()

            # # Convert pandas dataframe to PyTorch tensors
            # Convert units
            # (precip/1000)   # kg/m2/h = mm/h -> m/h
            # (pet/1000/3600) # kg/m2/h = mm/h -> m/s

            """Numpy implementation
            precip = np.array([self.forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm])
            pet = np.array([self.forcing_df["potential_evaporation"].values / cfg.conversions.m_to_mm/ cfg.conversions.hr_to_sec])
            """

            precip = torch.tensor(
                forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm,
                device=cfg.device,
            )
            pet = torch.tensor(
                forcing_df["potential_evaporation"].values
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
            output_tensor[i] = x_tr

        return output_tensor
        # # Creating a time interval
        # time_values = self.forcing_df["date"].values
        # self.timestep_map = {time: idx for idx, time in enumerate(time_values)}

    def get_observations(self, cfg: DictConfig):
        # # TODO FIND OBSERVATION DATA TO TRAIN AGAINST
        basin_ids = cfg.data.basin_id

        # Calculate the time difference between end_time and start_time
        time_difference = self.end_time - self.start_time

        # Calculate the total number of hours between the two datetimes (adding 1 for index error)
        hours_difference = int(time_difference.total_seconds() / 3600) + 1
        output_tensor = torch.zeros([len(basin_ids), hours_difference, 2])

        for i in tqdm(range(len(basin_ids)), desc="Reading observations"):
            id = basin_ids[i]
            obs_q_ = pd.read_csv(cfg.data.compare_results_file.format(id))
            obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
            self.obs_q = obs_q_[self.start_time : self.end_time].copy()

        """Numpy implementation
        self.y = self.obs_q['QObs(mm/h)'].values
        """

        self.n_timesteps = len(self.obs_q)
        return torch.tensor(self.obs_q["QObs(mm/h)"].values, device=cfg.device)

    def get_synthetic(self, cfg: DictConfig):
        # Define the file path
        dir_path = Path(cfg.synthetic.output_dir)
        file_path = dir_path / (cfg.synthetic.nams + ".npy")
        synthetic_q = np.load(file_path)
        self.obs_q = synthetic_q
        self.n_timesteps = len(self.obs_q)

        return torch.tensor(synthetic_q, device=cfg.device)

    def get_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        file_name = cfg.data.attributes_file
        basin_ids = cfg.data.basin_id
        # Load the txt data into a DataFrame
        data = pd.read_csv(file_name, sep=",")
        data["gauge_id"] = data["gauge_id"].str.replace("Gage-", "").str.zfill(8)
        # # Filter the DataFrame for the specified basin id
        filtered_data = data.loc[data["gauge_id"] == basin_ids]
        slope = filtered_data["slope_mean"].item()
        vcmx25 = filtered_data["vcmx25_mean"].item()
        mfsno = filtered_data["mfsno_mean"].item()
        cwpvt = filtered_data["cwpvt_mean"].item()
        # soil_depth = (
        #     filtered_data["soil_depth_statsgo"].item() * cfg.conversions.m_to_cm
        # )
        # soil_texture = filtered_data["soil_texture_class"].item()
        # soil_index = filtered_data["soil_index"].item()
        return torch.tensor([[slope, vcmx25, mfsno, cwpvt]])

    def get_cfe_params(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        cfe_params = dict()

        cfe_cfg = cfg.data

        # GET VALUES FROM CONFIGURATION FILE.
        cfe_params = {
            "catchment_area_km2": torch.tensor(
                [cfe_cfg.catchment_area_km2], dtype=torch.float
            ),
            "alpha_fc": torch.tensor([cfe_cfg.alpha_fc], dtype=torch.float),
            "soil_params": {
                "bb": torch.tensor([cfe_cfg.bb], dtype=torch.float),
                "smcmax": torch.tensor([cfe_cfg.smcmax], dtype=torch.float),
                "slop": torch.tensor([cfe_cfg.slop], dtype=torch.float),
                "D": torch.tensor([cfe_cfg.D], dtype=torch.float),
                "satpsi": torch.tensor([cfe_cfg.satpsi], dtype=torch.float),
                "wltsmc": torch.tensor([cfe_cfg.wltsmc], dtype=torch.float),
                "scheme": cfg.soil_scheme,
            },
            "max_gw_storage": torch.tensor([cfe_cfg.max_gw_storage], dtype=torch.float),
            "expon": torch.tensor([cfe_cfg.expon], dtype=torch.float),
            "Cgw": torch.tensor([cfe_cfg.Cgw], dtype=torch.float),
            "K_lf": torch.tensor([cfe_cfg.K_lf], dtype=torch.float),
            "K_nash": torch.tensor([cfe_cfg.K_nash], dtype=torch.float),
            "nash_storage": torch.tensor([cfe_cfg.nash_storage], dtype=torch.float),
            "giuh_ordinates": torch.tensor([cfe_cfg.giuh_ordinates], dtype=torch.float),
            "surface_partitioning_scheme": cfe_cfg.partition_scheme,
        }

        return cfe_params
