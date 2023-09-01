"""                                                                      
            dddddddd                                                                
            d::::::d       CCCCCCCCCCCCCFFFFFFFFFFFFFFFFFFFFFFEEEEEEEEEEEEEEEEEEEEEE
            d::::::d    CCC::::::::::::CF::::::::::::::::::::FE::::::::::::::::::::E
            d::::::d  CC:::::::::::::::CF::::::::::::::::::::FE::::::::::::::::::::E
            d:::::d  C:::::CCCCCCCC::::CFF::::::FFFFFFFFF::::FEE::::::EEEEEEEEE::::E
    ddddddddd:::::d C:::::C       CCCCCC  F:::::F       FFFFFF  E:::::E       EEEEEE
  dd::::::::::::::dC:::::C                F:::::F               E:::::E             
 d::::::::::::::::dC:::::C                F::::::FFFFFFFFFF     E::::::EEEEEEEEEE   
d:::::::ddddd:::::dC:::::C                F:::::::::::::::F     E:::::::::::::::E   
d::::::d    d:::::dC:::::C                F:::::::::::::::F     E:::::::::::::::E   
d:::::d     d:::::dC:::::C                F::::::FFFFFFFFFF     E::::::EEEEEEEEEE   
d:::::d     d:::::dC:::::C                F:::::F               E:::::E             
d:::::d     d:::::d C:::::C       CCCCCC  F:::::F               E:::::E       EEEEEE
d::::::ddddd::::::dd C:::::CCCCCCCC::::CFF:::::::FF           EE::::::EEEEEEEE:::::E
 d:::::::::::::::::d  CC:::::::::::::::CF::::::::FF           E::::::::::::::::::::E
  d:::::::::ddd::::d    CCC::::::::::::CF::::::::FF           E::::::::::::::::::::E
   ddddddddd   ddddd       CCCCCCCCCCCCCFFFFFFFFFFF           EEEEEEEEEEEEEEEEEEEEEE
"""

from omegaconf import DictConfig
import logging
import time
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from models.physics.bmi_cfe import BMI_CFE
import pandas as pd
import numpy as np
from utils.transform import normalization, to_physical
from models.MLP import MLP
from data.Data import Data

log = logging.getLogger("models.dCFE")


class dCFE(nn.Module):
    def __init__(self, cfg: DictConfig, Data) -> None:
        """

        :param cfg:
        """
        super(dCFE, self).__init__()
        self.cfg = cfg

        # Set up MLP instance
        self.normalized_c = normalization(Data.basin_attributes)
        self.MLP = MLP(self.cfg)

        # Instantiate the parameters you want to learn
        self.refkdt = torch.zeros([self.normalized_c.shape[0]])
        self.satdk = torch.zeros([self.normalized_c.shape[0]])
        # self.refkdt = nn.Parameter(torch.zeros([self.normalized_c.shape[0]]))
        # self.satdk = nn.Parameter(torch.zeros([self.normalized_c.shape[0]]))

        # Initialize the CFE model
        self.cfe_instance = BMI_CFE(
            refkdt=self.refkdt,
            satdk=self.satdk,
            cfg=self.cfg,
            cfe_params=Data.cfe_params,
        )
        self.cfe_instance.initialize()

    def forward(self, x):  # -> (Tensor, Tensor):
        """
        The forward function to model runoff through CFE model
        :param x: Precip and PET forcings (m/h)
        :return: runoff to be used for validation (mm/h)
        """
        # Read the forcing
        precip = x[0][0]
        pet = x[0][1]

        # Set precip and PET values in CFE
        self.cfe_instance.set_value(
            "atmosphere_water__time_integral_of_precipitation_mass_flux", precip
        )
        self.cfe_instance.set_value("water_potential_evaporation_flux", pet)

        # Run the model with the NN-trained parameters (refkdt and satdk)
        self.cfe_instance.update()

        # Get the runoff
        self.runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm

        return self.runoff

    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)

    def print(self):
        log.info(f"refkdt: {self.refkdt.tolist()[0]:.6f}")
        log.info(f"satdk: {self.satdk.tolist()[0]:.6f}")

    def mlp_forward(self) -> None:
        """
        A function to run MLP(). It sets the parameter values used within MC
        """
        self.refkdt, self.satdk = self.MLP(self.normalized_c)
