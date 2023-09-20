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
from models.fao_pet import FAO_PET
import json

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # Read in start and end datetime, Get the size of the observation
        self.start_time = datetime.strptime(
            cfg.data["start_time"], r"%Y-%m-%d %H:%M:%S"
        )
        self.end_time = datetime.strptime(cfg.data["end_time"], r"%Y-%m-%d %H:%M:%S")
        
        self.n_timesteps = self.calc_timestep_size(cfg)
        
        # Convert the basin ids to strings
        basin_ids = cfg.data.basin_ids
        self.basin_ids = [str(id) for id in basin_ids]
        
        self.x = self.get_forcings(cfg)

        self.c = self.get_dynamic_attributes(cfg)

        self.basin_attributes = self.get_static_attributes(cfg)

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
        # TODO: Check this
        return self.x[..., index, ...], self.y[..., index, ...]

    def __len__(self):
        """
        Method from the torch.Dataset parent class. Returns the number of timesteps
        """
        # TODO: Check this
        return self.x.shape[1]
      
    def calc_timestep_size(self, cfg: DictConfig):
        # Calculate the time difference between end_time and start_time
        time_difference = self.end_time - self.start_time

        # Calculate the total number of hours between the two datetimes
        # Currently hourly
        n_timesteps = int(time_difference.total_seconds() / 3600) + 1
        
        return n_timesteps

    def get_forcings(self, cfg: DictConfig):
      
        output_tensor = torch.zeros([len(self.basin_ids), self.n_timesteps, 2])
        
        # Read forcing data into pandas dataframe
        for i, basin_id in tqdm(enumerate(self.basin_ids), desc="Reading forcings"):
            forcing_df_ = pd.read_csv(cfg.data.forcing_file.format(basin_id))
            forcing_df_.set_index(pd.to_datetime(forcing_df_["date"]), inplace=True)
            forcing_df = forcing_df_[self.start_time : self.end_time].copy()

            # # Convert pandas dataframe to PyTorch tensors
            # Convert units
            # (precip/1000)   # kg/m2/h = mm/h -> m/h
            # (pet/1000/3600) # kg/m2/h = mm/h -> m/s

            precip = torch.tensor(
                forcing_df["total_precipitation"].values / cfg.conversions.m_to_mm,
                device=cfg.device,
            )
            _pet = FAO_PET(cfg=self.cfg, nldas_forcing=forcing_df, basin_id=basin_id).calc_PET()
            pet = torch.tensor(_pet.values, device=cfg.device)

            x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
            x_tr = x_.transpose(0, 1)
            output_tensor[i] = x_tr

        return output_tensor
      
        # # Creating a time interval
        # time_values = self.forcing_df["date"].values
        # self.timestep_map = {time: idx for idx, time in enumerate(time_values)}

    def get_observations(self, cfg: DictConfig):

        output_tensor = torch.zeros([len(self.basin_ids), self.n_timesteps, 1])

        for i, basin_id in tqdm(enumerate(self.basin_ids), desc="Reading observations"):
            obs_q_ = pd.read_csv(cfg.data.compare_results_file.format(basin_id))
            obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
            q = torch.tensor(obs_q_["QObs(mm/h)"][self.start_time : self.end_time].copy().values / cfg.conversions.m_to_mm, device=cfg.device) # TODO: Check unit conversion
            y_ = torch.stack([q])
            y_tr = y_.transpose(0, 1)
            output_tensor[i] = y_tr
        
        return output_tensor

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

        output_tensor = torch.zeros([len(self.basin_ids), self.n_timesteps, 3])
        
        # Read forcing data into pandas dataframe
        for i, basin_id in tqdm(enumerate(self.basin_ids), desc="Reading dynamic attributes"):
            forcing_df_ = pd.read_csv(cfg.data.forcing_file.format(basin_id))
            forcing_df_.set_index(pd.to_datetime(forcing_df_["date"]), inplace=True)
            forcing_df = forcing_df_[self.start_time : self.end_time].copy()

            Eo = torch.tensor(forcing_df["potential_energy"].values, device=cfg.device)
            cf = torch.tensor(forcing_df["convective_fraction"].values, device=cfg.device)
            R_l = torch.tensor(forcing_df["longwave_radiation"].values, device=cfg.device)

            c_ = torch.stack([Eo, cf, R_l])  # Index 0: Precip, index 1: PET
            c_tr = c_.transpose(0, 1)
            output_tensor[i] = c_tr

        return output_tensor

    def get_static_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """

        # Load the txt data into a DataFrame
        file_name = cfg.data.attributes_file
        data = pd.read_csv(file_name, sep=",")
        
        # Convert the gauge_id column to strings because the gauge_id has a Gage- prefix
        data["gauge_id"] = data["gauge_id"].str.replace("Gage-", "").str.zfill(8)
        basin_ids = data["gauge_id"].values

        # Filter the DataFrame for the specified basin id
        filtered_data = data.loc[data["gauge_id"].astype(str).isin(basin_ids)]
        slope = torch.tensor(filtered_data["slope_mean"].values, device=cfg.device,)
        vcmx25 = torch.tensor(filtered_data["vcmx25_mean"].values, device=cfg.device,)
        mfsno = torch.tensor(filtered_data["mfsno_mean"].values, device=cfg.device,)
        cwpvt = torch.tensor(filtered_data["cwpvt_mean"].values, device=cfg.device,)
       
        # Stack the tensors along a new dimension (dimension 1)
        return torch.stack([slope, vcmx25, mfsno, cwpvt])

    def get_cfe_params(self, cfg: DictConfig):
        """
        Reading attributes from the soil params JSON files based on basin_ids
        """
        cfe_params_list = []

        for basin_id in self.basin_ids:
            json_file_path = cfg.data.json_params_dir.format(basin_id[:8])

            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)

            cfe_params = {
                "catchment_area_km2": torch.tensor([json_data["catchment_area_km2"]], dtype=torch.float),
                "alpha_fc": torch.tensor([json_data["alpha_fc"]], dtype=torch.float),
                "soil_params": {
                    "bb": torch.tensor([json_data["soil_params"]["bb"]], dtype=torch.float),
                    "smcmax": torch.tensor([json_data["soil_params"]["smcmax"]], dtype=torch.float),
                    "slop": torch.tensor([json_data["soil_params"]["slop"]], dtype=torch.float),
                    "D": torch.tensor([json_data["soil_params"]["D"]], dtype=torch.float),
                    "satpsi": torch.tensor([json_data["soil_params"]["satpsi"]], dtype=torch.float),
                    "wltsmc": torch.tensor([json_data["soil_params"]["wltsmc"]], dtype=torch.float),
                    "scheme": cfg.soil_scheme,
                },
                "max_gw_storage": torch.tensor([json_data["max_gw_storage"]], dtype=torch.float),
                "expon": torch.tensor([json_data["expon"]], dtype=torch.float),
                "Cgw": torch.tensor([json_data["Cgw"]], dtype=torch.float),
                "K_lf": torch.tensor([json_data["K_lf"]], dtype=torch.float),
                "K_nash": torch.tensor([json_data["K_nash"]], dtype=torch.float),
                "nash_storage": torch.tensor(json_data["nash_storage"], dtype=torch.float),
                "giuh_ordinates": torch.tensor(json_data["giuh_ordinates"], dtype=torch.float),
                "surface_partitioning_scheme": json_data["partition_scheme"],
            }

            cfe_params_list.append(cfe_params)

        return cfe_params_list
      
    def create_GIUH_ordinates(self, original_giuh=[1.], max_GIUH_ordinate_size=10):
            """ Create max_GIUH_ordinate_size-by-1 GIUH ordinates
                max_GIUH_ordinate_size (int)
            """
            _giuh_ordinates = torch.tensor(original_giuh, dtype=torch.float)
            giuh_ordinates = torch.zeros((1, max_GIUH_ordinate_size), dtype=torch.float)
            # Fill in the giuh_ordinates values
            giuh_ordinates[0, :len(_giuh_ordinates)] = _giuh_ordinates
            return giuh_ordinates
