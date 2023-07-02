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
        # smcmax_ = torch.tensor([0.3]) # TODO: move to read_test_params() func later
        # self.smcmax = nn.ParameterList([])
        # for i in range(smcmax_.shape[0]):
        #     self.smcmax.append(nn.Parameter(smcmax_[i]))
        # self.smcmax = nn.Parameter(np.array([0.3]))
        self.smcmax = np.array([0.3])
            
        # self.c = generate_soil_metrics(self.cfg, self.smcmax)

        # Initialize the model 
        self.cfe_instance = BMI_CFE(
            self.cfg["src\data"],
            # self.c,
            self.smcmax
            ) #? Probably this is where the NN parameter fits? 
        # self.c necessary? No need? 
        self.cfe_instance.initialize()
        
        # self.slope = nn.Parameter(torch.tensor(0.0))
        # self.smcmax = nn.Parameter(torch.tensor(0.0))


    def forward(self, x): # -> (Tensor, Tensor):
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
        precip = x[0][0][0].numpy()
        pet = x[0][0][1].numpy()
        # precip = x[0][0][0]
        # pet = x[0][1]
        
        # Set precip and PET values 
        self.cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip)
        self.cfe_instance.set_value('water_potential_evaporation_flux', pet)
        
        # Run the model 
        self.cfe_instance.update()
        
        # Get the runoff 
        runoff = self.cfe_instance.return_runoff() * self.cfg.conversions.m_to_mm
        
        #? where cfe_instance.finalize() fits? 
        
        return runoff
    
    # def generate_soil_metrics(
    #         cfg: DictConfig,
    #         soils_df: pd.DataFrame,
    #         alpha: torch.nn.ParameterList,
    #         n: torch.nn.ParameterList,
    #     ) -> Tensor:
    #         """
    #         Reading the soils dataframe
    #         :param cfg: the config file
    #         :param wilting_point_psi_cm wilting point (the amount of water not available for plants or not accessible by plants)

    #         Below are the variables used inside of the soils dataframe:
    #         Texture: The soil classification
    #         theta_r: Residual Water Content
    #         theta_e: Wilting Point
    #         alpha(cm^-1): ???"
    #         n: ???
    #         m: ???
    #         Ks(cm/h): Saturated Hydraulic Conductivity

    #         :return:
    #         """
    #         h = torch.tensor(
    #             cfg.data.wilting_point_psi, device=cfg.device
    #         )  # Wilting point in cm
    #         initial_psi = torch.tensor(
    #             cfg.data.initial_psi, device=cfg.device
    #         )
    #         # alpha = torch.tensor(self.soils_df["alpha(cm^-1)"], device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    #         # n = torch.tensor(self.soils_df["n"], device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    #         # m = torch.tensor(self.soils_df["m"], device=cfg.device)  # TODO We're calculating this through n
    #         theta_e = torch.tensor(soils_df["theta_e"], device=cfg.device)
    #         theta_r = torch.tensor(soils_df["theta_r"], device=cfg.device)
    #         # k_sat = torch.tensor(self.soils_df["Ks(cm/h)"] device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    #         # ksat_cm_per_h = k_sat * cfg.constants.frozen_factor
    #         m = torch.zeros(len(alpha), device=cfg.device)
    #         theta_wp = torch.zeros(len(alpha), device=cfg.device)
    #         theta_init = torch.zeros(len(alpha), device=cfg.device)
    #         bc_lambda = torch.zeros(len(alpha), device=cfg.device)
    #         bc_psib_cm = torch.zeros(len(alpha), device=cfg.device)
    #         h_min_cm = torch.zeros(len(alpha), device=cfg.device)
    #         for i in range(len(alpha)):
    #             single_alpha = alpha[i]
    #             single_n = n[i]
    #             m[i] = calc_m(
    #                 single_n
    #             )  # Commenting out temporarily so that our test cases match
    #             theta_wp[i] = calc_theta_from_h(
    #                 h, single_alpha, single_n, m[i], theta_e[i], theta_r[i]
    #             )
    #             theta_init[i] = calc_theta_from_h(initial_psi, single_alpha, single_n, m[i], theta_e[i], theta_r[i])
    #             bc_lambda[i] = calc_bc_lambda(m[i])
    #             bc_psib_cm[i] = calc_bc_psib(single_alpha, m[i])
    #             h_min_cm[i] = calc_h_min_cm(bc_lambda[i], bc_psib_cm[i])

    #         soils_data = torch.stack(
    #             [
    #                 theta_r,
    #                 theta_e,
    #                 theta_wp,
    #                 theta_init,
    #                 m,
    #                 bc_lambda,
    #                 bc_psib_cm,
    #                 h_min_cm,
    #             ]
    #         )  # Putting all numeric columns in a tensor other than the Texture column
    #         return soils_data.transpose(0,1)
