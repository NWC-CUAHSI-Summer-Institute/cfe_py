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

from dpLGAR.data.utils import read_df

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.x = self.get_forcings(cfg)

        self.soil_attributes = self.get_polaris_atributes(cfg)

        self.y = self.get_observations(cfg)

        self.custom_normalization = {
            0: {"centering": "min", "scaling": "minmax"},
        }
        self._setup_normalization()

        self.normalized_soil_attributes = (
            self.soil_attributes - self.scaler["center"]
        ) / self.scaler["scale"]

    def _setup_normalization(self):
        """
        A function inspired by Neural Hydrology to normalize attributes
        https://github.com/neuralhydrology/neuralhydrology/blob/068135a44814796fa3ad3eb47656a00449fd1b1a/neuralhydrology/datasetzoo/basedataset.py#L721
        """
        # initialize scaler dict with default center and scale values (mean and std)
        self.scaler = {
            "center": self.soil_attributes.mean(dim=[0, 1], keepdim=True),
            "scale": self.soil_attributes.std(dim=[0, 1], keepdim=True),
        }

        # check for feature-wise custom normalization
        for feature_idx, feature_specs in self.custom_normalization.items():
            for key, val in feature_specs.items():
                # check for custom treatment of the center
                if key == "centering":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["center"][..., feature_idx] = 0.0
                    elif val.lower() == "median":
                        self.scaler["center"][..., feature_idx] = torch.median(
                            self.soil_attributes[..., feature_idx]
                        )
                    elif val.lower() == "min":
                        self.scaler["center"][..., feature_idx] = torch.min(
                            self.soil_attributes[..., feature_idx]
                        )
                    elif val.lower() == "mean":
                        # do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown centering method {val}")

                # check for custom treatment of the scale
                elif key == "scaling":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["scale"][..., feature_idx] = 1.0
                    elif val == "minmax":
                        self.scaler["scale"][..., feature_idx] = torch.max(
                            self.soil_attributes[..., feature_idx]
                        ) - torch.min(self.soil_attributes[..., feature_idx])
                    elif val == "std":
                        # do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown scaling method {val}")
                else:
                    # raise ValueError to point to the correct argument names
                    raise ValueError(
                        "Unknown dict key. Use 'centering' and/or 'scaling' for each feature."
                    )

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
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        file_path = cfg.data.forcing_file
        start_date = cfg.data.time_interval.warmup
        end_date = cfg.data.time_interval.end
        cols = ["date", "potential_evaporation", "total_precipitation"]
        data = pd.read_csv(file_path, usecols=cols, parse_dates=["date"])
        filtered_data = data.query("@start_date <= date <= @end_date")
        # Unit is kg/m^2
        precip = torch.tensor(
            filtered_data["total_precipitation"].to_numpy(), device=cfg.device
        )
        # Units is kg/m^2
        PET = torch.tensor(
            filtered_data["potential_evaporation"].to_numpy(), device=cfg.device
        )
        stacked_forcings = torch.stack([precip, PET])
        # Note: kg/m^2 == mm, so we need to convert to CM
        x = stacked_forcings.transpose(0, 1) * cfg.conversions.mm_to_cm
        return x

    def to_lgar_c_format(self, df, file_name):
        """
        For converting filtered data to LGAR-C format
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime("%-m/%-d/%y %H:%M")
        df = df.rename(
            columns={
                "date": "Time",
                "potential_evaporation": "PET(mm/h)",
                "total_precipitation": "P(mm/h)",
            }
        )
        df = df[["Time", "P(mm/h)", "PET(mm/h)"]]
        df["PET(mm/h)"] = df["PET(mm/h)"].clip(lower=0)
        df.to_csv(file_name, index=False)

    def get_basin_area(self, cfg):
        """
        Read and filter a CSV file for a specified basin id to get the basin area.
        :param cfg: the DictConfig obj

        :return: Basin area for the specified basin id.
        """
        file_path = cfg.data.area_file
        basin_id = cfg.basin_id
        data = pd.read_csv(file_path)
        formatted_basin_id = f"Gage-{basin_id}"
        filtered_data = data[data["gauge_id"] == formatted_basin_id]
        return filtered_data["AREA_sqkm"].values[0] if not filtered_data.empty else None

    def get_camels_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        file_name = cfg.data.attributes_file
        basin_id = cfg.basin_id
        # Load the txt data into a DataFrame
        data = pd.read_csv(file_name, sep=";")
        data["gauge_id"] = data["gauge_id"].astype("str").str.zfill(8)
        # Filter the DataFrame for the specified basin id
        filtered_data = data[data["gauge_id"] == basin_id]
        soil_depth = (
            filtered_data["soil_depth_statsgo"].item() * cfg.conversions.m_to_cm
        )
        soil_texture = filtered_data["soil_texture_class"].item()
        soil_index = filtered_data["soil_index"].item()
        return [soil_depth, soil_texture, soil_index]

    def get_polaris_atributes(self, cfg: DictConfig):
        file_name = cfg.data.attributes_file
        # Load the txt data into a DataFrame
        df = pd.read_csv(file_name)

        # Filter columns for soil %, Ph, and organic_matter
        clay_columns = [col for col in df.columns if col.startswith("clay")]
        sand_columns = [col for col in df.columns if col.startswith("sand")]
        silt_columns = [col for col in df.columns if col.startswith("silt")]
        ph_columns = [col for col in df.columns if col.startswith("ph")]
        organic_matter_columns = [col for col in df.columns if col.startswith("om")]

        # Create a numpy array from the columns
        clay_data = df[clay_columns].values
        sand_data = df[sand_columns].values
        silt_data = df[silt_columns].values
        ph_data = df[ph_columns].values
        organic_matter_data = df[organic_matter_columns].values

        # Shape (<num_points>, <num_layers>, <num_attributes>)
        soil_attributes = torch.stack(
            [
                torch.from_numpy(clay_data),
                torch.from_numpy(sand_data),
                torch.from_numpy(silt_data),
                torch.from_numpy(ph_data),
                torch.from_numpy(organic_matter_data),
            ],
            dim=-1,
        )

        return soil_attributes

    def get_observations(self, cfg: DictConfig):
        """
        reading observations from NLDAS forcings
        :param cfg: the DictConfig obj
        """
        obs = read_df(cfg.data.observations_file)
        precip = obs["QObs(mm/h)"]
        precip_tensor = torch.tensor(precip.to_numpy(), device=cfg.device)
        nan_mask = torch.isnan(precip_tensor)
        # Filling NaNs with 0 as there is no streamflow
        precip_tensor[nan_mask] = 0.0
        return precip_tensor

    def _plot_normalization(self):
        import matplotlib.pyplot as plt

        # Plotting each attribute
        attributes = ["clay", "sand", "silt", "ph", "om"]
        for i, attribute in enumerate(attributes):
            plt.figure()
            attribute_values = self.normalized_soil_attributes[..., i].flatten()
            plt.scatter(range(len(attribute_values)), attribute_values)
            plt.title(f"Normalized {attribute}")
            plt.show()
