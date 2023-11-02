"""   _  _____ ______ ______ 
     | |/ ____|  ____|  ____|
   __| | |    | |__  | |__   
  / _` | |    |  __| |  __|  
 | (_| | |____| |    | |____ 
  \__,_|\_____|_|    |______|
"""

from omegaconf import DictConfig
import logging
import time
from tqdm import tqdm
import torch

torch.set_default_dtype(torch.float64)
from torch import Tensor
import torch.nn as nn
from models.physics.bmi_cfe import BMI_CFE
import pandas as pd
import numpy as np
from utils.transform import normalization, to_physical
from models.MLP import MLP
from data.Data import Data

log = logging.getLogger("models.dCFE")


class SyntheticCFE(nn.Module):
    def __init__(self, cfg: DictConfig, Data) -> None:
        """

        :param cfg:
        """
        super(SyntheticCFE, self).__init__()

        # CFE parameters are in this cfg[src\data]
        self.cfg = cfg

        # Instantiate the params you want to initialize with
        num_basins = len(self.cfg.data.basin_ids)
        self.refkdt = self.cfg.synthetic.refkdt * torch.ones((num_basins))
        self.satdk = self.cfg.synthetic.satdk * torch.ones((num_basins))

        # Initializing Values
        self.c = None

        # def cfe_initialize(self):
        # Initialize the model
        self.cfe_instance = BMI_CFE(
            refkdt=self.refkdt, satdk=self.satdk, cfg=self.cfg, cfe_params=Data.params
        )

        self.cfe_instance.initialize()

    def forward(self, x):  # -> (Tensor, Tensor):
        """
        The forward function to model runoff through CFE model
        :param x: Precip and PET forcings (m/h)
        :return: runoff to be used for validation (mm/h)
        """

        # Read the forcing
        precip = x[:, :, 0]
        pet = x[:, :, 1]

        # Set precip and PET values
        self.cfe_instance.set_value(
            "atmosphere_water__time_integral_of_precipitation_mass_flux", precip
        )
        self.cfe_instance.set_value("water_potential_evaporation_flux", pet)

        # Run the model
        self.cfe_instance.update()

        # Get the runoff
        self.runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm

        return self.runoff

    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)
