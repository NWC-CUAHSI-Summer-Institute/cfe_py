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
from models.fao_pet import FAO_PET

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # Read in start and end datetime
        self.start_time = datetime.strptime(
            cfg.data["start_time"], r"%Y-%m-%d %H:%M:%S"
        )
        self.end_time = datetime.strptime(cfg.data["end_time"], r"%Y-%m-%d %H:%M:%S")

        self.x = self.get_forcings(cfg)

        self.c = self.get_dynamic_attributes(cfg)

        self.basin_attributes = self.get_static_attributes(cfg)

        if (cfg.run_type == "ML") | (cfg.run_type == "generate_synthetic"):
            self.y = self.get_observations(cfg)
        elif cfg.run_type == "ML_synthetic_test":
            self.y = self.get_synthetic(cfg)

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
        """
        Reading NLDAS forcing data for a watershed 
        """

        # Read NLDAS forcing data into pandas dataframe
        forcing_df_ = pd.read_csv(cfg.data["forcing_file"])
        forcing_df_.set_index(pd.to_datetime(forcing_df_["date"]), inplace=True)
        forcing_df = forcing_df_[self.start_time : self.end_time].copy()
        self.forcing_df = forcing_df

        # # Convert pandas dataframe to PyTorch tensors
        # Precipitation
        # Unit conversion (precip/1000) : kg/m2/h = mm/h -> m/h
        precip = torch.tensor(
            forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm,
            device=cfg.device,
        )

        # PET calculation based on NLDAS forcing 
        # (pet/1000/3600) # kg/m2/h = mm/h -> m/s
        _pet = FAO_PET(cfg=self.cfg, nldas_forcing=forcing_df).calc_PET()
        pet = torch.tensor(_pet.values, device=cfg.device)

        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        x_tr = x_.transpose(0, 1)

        return x_tr


    def get_observations(self, cfg: DictConfig):
        """
        Reading observed streamflow timeseries generated for the watershed 
        """
        obs_q_ = pd.read_csv(cfg.data["compare_results_file"])
        obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
        self.obs_q = obs_q_[self.start_time : self.end_time].copy()
        self.n_timesteps = len(self.obs_q)
        return torch.tensor(self.obs_q["QObs(mm/h)"].values, device=cfg.device)

    def get_synthetic(self, cfg: DictConfig):
        """
        Reading synthetic streamflow timeseries generated for the watershed 
        """
        # Define the file path
        dir_path = Path(cfg.synthetic.output_dir)
        file_path = dir_path / cfg.synthetic.nams
        synthetic_q = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.obs_q = synthetic_q[self.start_time : self.end_time].copy()
        self.n_timesteps = len(self.obs_q)

        return torch.tensor(self.obs_q.y_hat, device=cfg.device)

    def get_dynamic_attributes(self, cfg:DictConfig):
        Eo = torch.tensor(self.forcing_df["potential_energy"].values, device=cfg.device)
        cf = torch.tensor(self.forcing_df["convective_fraction"].values, device=cfg.device)
        R_l = torch.tensor(self.forcing_df["longwave_radiation"].values, device=cfg.device)
        c_ = torch.stack([Eo, cf, R_l])  # Index 0: Poential evaporation, index 1: convective fraction 
        c_tr = c_.transpose(0, 1)
        return c_tr

    def get_static_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        file_name = cfg.data.attributes_file
        basin_id = cfg.data.basin_id
        
        # Load the txt data into a DataFrame
        data = pd.read_csv(file_name, sep=",")
        data["gauge_id"] = data["gauge_id"].str.replace("Gage-", "").str.zfill(8)
        
        # # Filter the DataFrame for the specified basin id
        filtered_data = data[data["gauge_id"] == basin_id]
        slope = filtered_data["slope_mean"].item()
        vcmx25 = filtered_data["vcmx25_mean"].item()
        mfsno = filtered_data["mfsno_mean"].item()
        cwpvt = filtered_data["cwpvt_mean"].item()

        return torch.tensor([[slope, vcmx25, mfsno, cwpvt]])

    def get_cfe_params(self, cfg: DictConfig):
        """
        Reading CFE parameters
        """
        cfe_params = dict()

        # Create 10-by-1 GIUH ordinates
        giuh_ordinates = self.create_GIUH_ordinates(original_giuh=cfg.data.giuh_ordinates, max_GIUH_ordinate_size = 10)

        # GET VALUES FROM CONFIGURATION FILE.
        cfe_params = {
            "catchment_area_km2": torch.tensor(
                [cfg.data.catchment_area_km2], dtype=torch.float
            ),
            "alpha_fc": torch.tensor([cfg.data.alpha_fc], dtype=torch.float),
            "soil_params": {
                "bb": torch.tensor([cfg.data.bb], dtype=torch.float),
                "smcmax": torch.tensor([cfg.data.smcmax], dtype=torch.float),
                "slop": torch.tensor([cfg.data.slop], dtype=torch.float),
                "D": torch.tensor([cfg.data.D], dtype=torch.float),
                "satpsi": torch.tensor([cfg.data.satpsi], dtype=torch.float),
                "wltsmc": torch.tensor([cfg.data.wltsmc], dtype=torch.float),
                "scheme": cfg.soil_scheme,
            },
            "max_gw_storage": torch.tensor(
                [cfg.data.max_gw_storage], dtype=torch.float
            ),
            "expon": torch.tensor([cfg.data.expon], dtype=torch.float),
            "Cgw": torch.tensor([cfg.data.Cgw], dtype=torch.float),
            "K_lf": torch.tensor([cfg.data.K_lf], dtype=torch.float),
            "K_nash": torch.tensor([cfg.data.K_nash], dtype=torch.float),
            "nash_storage": torch.tensor([cfg.data.nash_storage], dtype=torch.float),
            "giuh_ordinates": giuh_ordinates,
            "surface_partitioning_scheme": cfg.data.partition_scheme,
        }

        return cfe_params


    def create_GIUH_ordinates(self, original_giuh=[1.], max_GIUH_ordinate_size=10):
        """ Create max_GIUH_ordinate_size-by-1 GIUH ordinates
            max_GIUH_ordinate_size (int)
        """
        _giuh_ordinates = torch.tensor(original_giuh, dtype=torch.float)
        giuh_ordinates = torch.zeros((1, max_GIUH_ordinate_size), dtype=torch.float)
        # Fill in the giuh_ordinates values
        giuh_ordinates[0, :len(_giuh_ordinates)] = _giuh_ordinates
        return giuh_ordinates