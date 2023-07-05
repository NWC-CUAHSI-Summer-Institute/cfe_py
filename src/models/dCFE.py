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
from src.models.physics.bmi_cfe import BMI_CFE
import pandas as pd
import numpy as np


log = logging.getLogger("models.dCFE")

class dCFE(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        """
        super(dCFE, self).__init__()

        self.cfg = cfg

        # Setting NN parameters
        parameters = {
            'bb': 5,
            'smcmax': 0.5,
            'satdk': 0.00001,
            'slop': 1,
            'max_gw_storage': 0.5,
            'expon': 7,
            'Cgw': 1,
            'K_lf': 0.5,
            'K_nash': 0.3
        }

        self.c = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(value, dtype=torch.float))
            for key, value in parameters.items()
        })
            
        """Numpy implementation
        self.smcmax = np.array([0.3])
        """

        # Initialize the model 
        self.cfe_instance = BMI_CFE(
            self.cfg["src\data"],
            c=self.c
            )
        
        # self.c necessary? No need? 
        self.cfe_instance.initialize()


    def forward(self, x): # -> (Tensor, Tensor):
        """
        The forward function to model runoff through CFE model 
        :param x: Precip and PET forcings (m/h)
        :return: runoff to be used for validation (mm/h)
        """
        # TODO implement the CFE functions
        
        # Read the forcing        
        """Numpy implementation
        precip = x[0][0][0].numpy()
        pet = x[0][0][1].numpy()
        """
        precip = x[0][0]
        pet = x[0][1]
        
        # Set precip and PET values 
        self.cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip)
        self.cfe_instance.set_value('water_potential_evaporation_flux', pet)
        
        # Run the model 
        self.cfe_instance.update()
        
        # Get the runoff 
        self.runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm
        
        return self.runoff
    
    def finalize(self):
        self.cfe_instance.finalize(print_mass_balance=True)
        
