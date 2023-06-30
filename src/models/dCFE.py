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


log = logging.getLogger("models.dCFE")

class dCFE(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        """
        super(dCFE, self).__init__()

        self.cfg = cfg

        # Setting NN parameters
        smcmax_ = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0]) # TODO: move to read_test_params() func later
        self.smcmax = nn.ParameterList([])
        for i in range(smcmax_.shape[0]):
            self.smcmax.append(nn.Parameter(smcmax_[i]))
            
        # self.c = generate_soil_metrics(self.cfg, self.smcmax)

        # Initialize the model 
        self.cfe_instance = BMI_CFE(
            self.cfg.data,
            self.c,
            self.smcmax
            ) #? Probably this is where the NN parameter fits? 
        # self.c necessary? No need? 
        self.cfe_instance.initialize()
        
        # self.slope = nn.Parameter(torch.tensor(0.0))
        # self.smcmax = nn.Parameter(torch.tensor(0.0))


    def forward(self, x) -> (Tensor, Tensor):
        """
        The forward function to model runoff through CFE model 
        
        The forward function to model Precip/PET through LGAR functions
        /* Note unit conversion:
        P and PET are rates (fluxes) in m/h
        
        # Pr [mm/h] * 1h/3600sec = Pr [mm/3600sec]
        # Model timestep (dt) = 300 sec (5 minutes for example)
        # convert rate to amount
        # Pr [mm/3600sec] * dt [300 sec] = Pr[mm] * 300/3600.

        runoff is in mm/h
        land_surface_water__runoff_depth[m/h] * 1000mm/m = runoff [mm/h]
        
        :param x: Precip and PET forcings
        :return: runoff to be used for validation
        """
        # TODO implement the CFE functions
        
        # Read the forcing         
        precip = x[0][0]
        pet = x[0][1]
        
        # Set precip and PET values 
        self.cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip)
        self.cfe_instance.set_value('water_potential_evaporation_flux', pet)
        
        # Run the model 
        self.cfe_instance.update()
        
        # Get the runoff 
        runoff = self.cfe_instance.get_value('land_surface_water__runoff_depth') * self.cfg.conversions.m_to_mm
        
        #? where cfe_instance.finalize() fits? 
        
        return runoff
