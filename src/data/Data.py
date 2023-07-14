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

        # Read in start and end datetime
        self.start_time = datetime.strptime(
            cfg["src\data"]["start_time"], r"%Y-%m-%d %H:%M:%S"
        )

        self.end_time = datetime.strptime(
            cfg["src\data"]["end_time"], r"%Y-%m-%d %H:%M:%S"
        )

        self.x = self.get_forcings(cfg)

        self.basin_attributes = self.get_attributes(cfg)

        self.y = self.get_observations(cfg)

        self.cfe_params = self.get_cfe_params(cfg)

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

    def get_forcings(self, cfg: DictConfig):
        # Read forcing data into pandas dataframe
        forcing_df_ = pd.read_csv(cfg["src\data"]["forcing_file"])
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
        return x_tr

        # # Creating a time interval
        # time_values = self.forcing_df["date"].values
        # self.timestep_map = {time: idx for idx, time in enumerate(time_values)}

    def get_observations(self, cfg: DictConfig):
        # # TODO FIND OBSERVATION DATA TO TRAIN AGAINST
        obs_q_ = pd.read_csv(cfg["src\data"]["compare_results_file"])
        obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
        self.obs_q = obs_q_[self.start_time : self.end_time].copy()

        """Numpy implementation
        self.y = self.obs_q['QObs(mm/h)'].values
        """

        self.n_timesteps = len(self.obs_q)
        return torch.tensor(self.obs_q["QObs(mm/h)"].values, device=cfg.device)

    def get_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        file_name = cfg["src\data"].attributes_file
        basin_id = cfg["src\data"].basin_id
        # Load the txt data into a DataFrame
        data = pd.read_csv(file_name, sep=",")
        data["gauge_id"] = data["gauge_id"].str.replace("Gage-", "")
        # # Filter the DataFrame for the specified basin id
        filtered_data = data[data["gauge_id"] == basin_id]
        slope = filtered_data["slope_mean"].item()
        vcmx25 = filtered_data["vcmx25_mean"].item()
        mfsno = filtered_data["mfsno_mean"].item()
        cwpvt = filtered_data["cwpvt_mean"].item()
        # soil_depth = (
        #     filtered_data["soil_depth_statsgo"].item() * cfg.conversions.m_to_cm
        # )
        # soil_texture = filtered_data["soil_texture_class"].item()
        # soil_index = filtered_data["soil_index"].item()
        return [slope, vcmx25, mfsno, cwpvt]  # No Lat and lon?

    def get_cfe_params(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        cfe_params = dict()

        cfe_cfg = cfg["src\data"]

        # GET VALUES FROM CONFIGURATION FILE.
        cfe_params["catchment_area_km2"] = torch.tensor(
            cfe_cfg.catchment_area_km2, dtype=torch.float
        )

        # Soil parameters
        cfe_params["alpha_fc"] = torch.tensor(cfe_cfg.alpha_fc, dtype=torch.float)
        cfe_params["soil_params"] = {}
        cfe_params["soil_params"]["bb"] = torch.tensor(cfe_cfg.bb, dtype=torch.float)
        cfe_params["soil_params"]["smcmax"] = torch.tensor(
            cfe_cfg.smcmax, dtype=torch.float
        )

        ####  Pass NN param later ####
        cfe_params["soil_params"]["slop"] = torch.tensor(
            cfe_cfg.slop, dtype=torch.float
        )
        cfe_params["soil_params"]["D"] = torch.tensor(cfe_cfg.D, dtype=torch.float)
        cfe_params["soil_params"]["satpsi"] = torch.tensor(
            cfe_cfg.satpsi, dtype=torch.float
        )
        cfe_params["soil_params"]["wltsmc"] = torch.tensor(
            cfe_cfg.wltsmc, dtype=torch.float
        )

        # Groundwater storage
        cfe_params["max_gw_storage"] = torch.tensor(
            cfe_cfg.max_gw_storage, dtype=torch.float
        )
        cfe_params["expon"] = torch.tensor(cfe_cfg.expon, dtype=torch.float)
        cfe_params["Cgw"] = torch.tensor(cfe_cfg.Cgw, dtype=torch.float)

        # Lateral flow
        cfe_params["K_lf"] = torch.tensor(cfe_cfg.K_lf, dtype=torch.float)
        cfe_params["K_nash"] = torch.tensor(cfe_cfg.K_nash, dtype=torch.float)
        cfe_params["nash_storage"] = torch.tensor(
            cfe_cfg.nash_storage, dtype=torch.float
        )

        # Routing
        cfe_params["giuh_ordinates"] = torch.tensor(
            cfe_cfg.giuh_ordinates, dtype=torch.float
        )

        # Partitioning parameters
        cfe_params["surface_partitioning_scheme"] = cfe_cfg.partition_scheme
        cfe_params["soil_params"]["scheme"] = cfe_cfg.soil_scheme

        # Other
        cfe_params["stand_alone"] = 0

        return cfe_params
