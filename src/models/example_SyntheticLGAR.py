"""

       _|            _|          _|_|_|    _|_|    _|_|_|
   _|_|_|  _|_|_|    _|        _|        _|    _|  _|    _|
 _|    _|  _|    _|  _|        _|  _|_|  _|_|_|_|  _|_|_|
 _|    _|  _|    _|  _|        _|    _|  _|    _|  _|    _|
   _|_|_|  _|_|_|    _|_|_|_|    _|_|_|  _|    _|  _|    _|
           _|
           _|
"""
from omegaconf import DictConfig
import logging
import torch
from torch import Tensor
import torch.nn as nn

from dpLGAR.data.utils import generate_soil_metrics
from dpLGAR.models.MLP import MLP
from dpLGAR.models.physics.GlobalParams import GlobalParams
from dpLGAR.models.physics.layers.Layer import Layer
from dpLGAR.models.physics.lgar.frozen_factor import (
    frozen_factor_hydraulic_conductivity,
)
from dpLGAR.models.physics.lgar.giuh import calc_giuh

log = logging.getLogger("models.dpLGAR")


class SyntheticLGAR(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        :param soil_information: soil attributes
        """
        super(SyntheticLGAR, self).__init__()

        self.cfg = cfg
        self.rank = cfg.local_rank

        self.alpha = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )
        self.n = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )
        self.ksat = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )
        self.max_ponded_depth = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )
        self.theta_e = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )
        self.theta_r = torch.zeros(
            [len(self.cfg.data.num_soil_layers)], device=self.cfg.device
        )

        # Initializing Values
        self.c = None
        self.cfg.data.soil_index = {
            "theta_r": 0,
            "theta_e": 1,
            "theta_wp": 2,
            "theta_init": 3,
            "m": 4,
            "bc_lambda": 5,
            "bc_psib_cm": 6,
            "h_min_cm": 7,
            "alpha": 8,
            "n": 9,
        }

        self.top_layer = None
        self.bottom_layer = None
        self.num_wetting_fronts = None
        self.wf_free_drainage_demand = None
        self.ending_volume = None
        self.precip = None
        self.PET = None
        self.AET = None
        self.ponded_water = None
        self.previous_precip = None
        self.infiltration = None
        self.runoff = None
        self.giuh_runoff = None
        self.giuh_runoff_queue = None
        self.discharge = None
        self.groundwater_discharge = None
        self.percolation = None

        self.global_params = None

        # Setting internal states (soil layers)
        self.set_internal_states()

    def set_internal_states(self):
        # Creating static soil params
        (
            self.alpha,
            self.n,
            self.ksat,
            self.ponded_max_depth,
            self.theta_e,
            self.theta_r,
        ) = self.mlp(self.normalized_soil_attributes)

        self.c = generate_soil_metrics(
            self.cfg, self.alpha, self.n, self.theta_e, self.theta_r
        )

        self.global_params = GlobalParams(self.cfg, self.ponded_depth_max)

        # Creating initial soil layer stack
        # We're only saving a reference to the top layer as all precip, PET, and runoff deal with it
        layer_index = 0  # This is the top layer
        self.top_layer = Layer(
            self.global_params,
            layer_index,
            self.c,
            self.ksat,
            self.rank,
        )

        # Gaining a reference to the bottom layer
        self.bottom_layer = self.top_layer
        while self.bottom_layer.next_layer is not None:
            self.bottom_layer = self.bottom_layer.next_layer

        # Determining the number of wetting fronts total
        self.num_wetting_fronts = self.calc_num_wetting_fronts()
        self.wf_free_drainage_demand = None

        # Running the initial mass balance check
        self.ending_volume = self.calc_mass_balance()

        # Setting output tracking params
        self.precip = torch.tensor(0.0, device=self.cfg.device)
        self.PET = torch.tensor(0.0, device=self.cfg.device)
        self.AET = torch.tensor(0.0, device=self.cfg.device)

        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficial wetting front
        self.ponded_water = torch.tensor(0.0, device=self.cfg.device)
        self.previous_precip = torch.tensor(0.0, device=self.cfg.device)
        self.infiltration = torch.tensor(0.0, device=self.cfg.device)
        self.runoff = torch.tensor(0.0, device=self.cfg.device)
        self.giuh_runoff = torch.tensor(0.0, device=self.cfg.device)
        self.giuh_runoff_queue = torch.zeros(
            [self.global_params.num_giuh_ordinates], device=self.cfg.device
        )
        self.discharge = torch.tensor(0.0, device=self.cfg.device)
        self.groundwater_discharge = torch.tensor(0.0, device=self.cfg.device)
        self.percolation = torch.tensor(0.0, device=self.cfg.device)
        # self.top_layer.print()

    def update_soil_parameters(self):
        self.global_params.ponded_depth_max = self.ponded_depth_max.clone()
        self.c = generate_soil_metrics(self.cfg, self.soils_df, self.alpha, self.n)
        self.top_layer.update_soil_parameters(self.c)

    def forward(self, i, x) -> (Tensor, Tensor):
        """
        The forward function to model Precip/PET through LGAR functions
        /* Note unit conversion:
        Pr and PET are rates (fluxes) in mm/h
        Pr [mm/h] * 1h/3600sec = Pr [mm/3600sec]
        Model timestep (dt) = 300 sec (5 minutes for example)
        convert rate to amount
        Pr [mm/3600sec] * dt [300 sec] = Pr[mm] * 300/3600.
        in the code below, subtimestep_h is this 300/3600 factor (see initialize from config in lgar.cxx)
        :param i: The current timestep index
        :param x: Precip and PET forcings
        :return: runoff to be used for validation
        """
        precip = x[0]
        pet = x[1]
        groundwater_discharge_sub = torch.tensor(0.0, device=self.cfg.device)
        bottom_boundary_flux = torch.tensor(0.0, device=self.cfg.device)
        ending_volume_sub = self.ending_volume.clone()
        if self.global_params.sft_coupled:
            frozen_factor_hydraulic_conductivity()
        subtimestep_h = self.cfg.models.subcycle_length_h
        for _ in range(int(self.cfg.models.num_subcycles)):
            self.top_layer.copy_states()
            precip_sub = precip * subtimestep_h
            # Making sure PET >=0
            pet_sub = torch.clamp(pet * subtimestep_h, min=0.0)
            previous_precip_sub = self.previous_precip.clone()
            ponded_depth_sub = precip_sub + self.ponded_water
            ponded_water_sub = torch.tensor(0.0, device=self.cfg.device)
            percolation_sub = torch.tensor(0.0, device=self.cfg.device)
            runoff_sub = torch.tensor(0.0, device=self.cfg.device)
            infiltration_sub = torch.tensor(0.0, device=self.cfg.device)
            AET_sub = torch.tensor(0.0, device=self.cfg.device)
            # Determining wetting cases
            create_surficial_front = self.create_surficial_front(
                previous_precip_sub, precip_sub
            )
            self.wf_free_drainage_demand = self.calc_wetting_front_free_drainage()
            self.top_layer.set_wf_free_drainage_demand(self.wf_free_drainage_demand)
            is_top_layer_saturated = self.top_layer.is_saturated()
            if pet > 0.0:
                AET_sub = self.top_layer.calc_aet(pet, subtimestep_h)
            self.precip = self.precip + precip_sub
            self.PET = self.PET + torch.max(pet_sub, torch.tensor(0.0))
            starting_volume_sub = self.calc_mass_balance()
            if create_surficial_front and is_top_layer_saturated is False:
                # ------------------------------------------------------------------------------------------------------
                # /* create a new wetting front if the following is true. Meaning there is no
                #    wetting front in the top layer to accept the water, must create one. */
                temp_pd = torch.tensor(0.0, device=self.cfg.device)
                # // move the wetting fronts without adding any water; this is done to close the mass balance
                temp_pd, AET_sub = self.move_wetting_front(
                    temp_pd,
                    AET_sub,
                    ending_volume_sub,
                    subtimestep_h,
                    bottom_boundary_flux,
                )
                # depth of the surficial front to be created
                dry_depth = self.top_layer.calc_dry_depth(subtimestep_h)
                (
                    ponded_depth_sub,
                    infiltration_sub,
                ) = self.top_layer.create_surficial_front(
                    dry_depth, ponded_depth_sub, infiltration_sub
                )
                self.top_layer.copy_states()
                self.infiltration = self.infiltration + infiltration_sub
            if create_surficial_front is False and ponded_depth_sub > 0:
                # ------------------------------------------------------------------------------------------------------
                # /* infiltrate water based on the infiltration capacity given no new wetting front
                #    is created and that there is water on the surface (or raining). */
                (
                    runoff_sub,
                    infiltration_sub,
                    ponded_depth_sub,
                ) = self.top_layer.insert_water(
                    subtimestep_h,
                    precip_sub,
                    ponded_depth_sub,
                    infiltration_sub,
                )

                self.infiltration = self.infiltration + infiltration_sub
                self.runoff = self.runoff + runoff_sub
                percolation_sub = infiltration_sub  # this gets updated later, probably not needed here

                ponded_water_sub = ponded_depth_sub
            else:
                (
                    ponded_depth_sub,
                    ponded_water_sub,
                    runoff_sub,
                ) = self.update_ponded_depth(ponded_depth_sub)
            if create_surficial_front is False:
                # ------------------------------------------------------------------------------------------------------
                # /* move wetting fronts if no new wetting front is created. Otherwise, movement
                #    of wetting fronts has already happened at the time of creating surficial front,
                #    so no need to move them here. */
                infiltration_temp = infiltration_sub.clone()
                infiltration_sub, AET_sub = self.move_wetting_front(
                    infiltration_sub,
                    AET_sub,
                    ending_volume_sub,
                    subtimestep_h,
                    bottom_boundary_flux,
                )
                percolation_sub = infiltration_sub.clone()
                self.percolation = (
                    self.percolation + percolation_sub
                )  # Make sure the values aren't getting lost
                infiltration_sub = infiltration_temp

            # ----------------------------------------------------------------------------------------------------------
            # calculate derivative (dz/dt) for all wetting fronts
            self.top_layer.calc_dzdt(ponded_depth_sub)
            ending_volume_sub = self.calc_mass_balance()
            self.previous_precip = precip_sub
            self.ending_volume = ending_volume_sub

            # ----------------------------------------------------------------------------------------------------------
            # mass balance at the subtimestep (local mass balance)
            local_mb = (
                starting_volume_sub
                + precip_sub
                + self.ponded_water
                - runoff_sub
                - AET_sub
                - ponded_water_sub
                - percolation_sub
                - ending_volume_sub
            )
            self.AET = self.AET + AET_sub
            self.ponded_water = ponded_water_sub

            # ----------------------------------------------------------------------------------------------------------
            # compute giuh runoff for the subtimestep if there is runoff or runoff recently
            if self.giuh_runoff_queue.sum() > 0 or runoff_sub > 0:
                giuh_runoff_sub, self.giuh_runoff_queue = calc_giuh(
                    self.global_params, self.giuh_runoff_queue, runoff_sub
                )
                self.giuh_runoff = self.giuh_runoff + giuh_runoff_sub
                self.discharge = self.discharge + giuh_runoff_sub
                self.groundwater_discharge = groundwater_discharge_sub
        # self.print_local_mass_balance(
        #     local_mb,
        #     starting_volume_sub,
        #     precip_sub,
        #     runoff_sub,
        #     AET_sub,
        #     ponded_water_sub,
        #     percolation_sub,
        #     ending_volume_sub,
        #     infiltration_sub,
        #     i,
        # )
        return self.runoff

    def calc_mass_balance(self) -> Tensor:
        """
        Calculates a mass balance from your variables (Known as lgar_calc_mass_bal() in the C code)
        This is a recursive stack function. It calls the top of the stack,
        and will go until the bottom is reach
        :return: Sum
        """
        return self.top_layer.mass_balance()

    def create_surficial_front(self, previous_precip, precip_sub):
        """
        Checks the volume of water on the surface, and if it's raining or has recently rained
        to determine if any water has infiltrated the surface
        :param previous_precip:
        :param precip_subtimestep:
        :param volon_timestep_cm:
        :return:
        """
        # This enusures we don't add extra mass from a previous storm event
        has_previous_precip = (previous_precip == 0.0).item()
        is_it_raining = (precip_sub > 0.0).item()
        is_there_ponded_water = (self.ponded_water == 0).item()
        return has_previous_precip and is_it_raining and is_there_ponded_water

    def calc_num_wetting_fronts(self):
        return self.top_layer.calc_num_wetting_fronts()

    def calc_wetting_front_free_drainage(self):
        """
        A function to determine the bottom-most layer impacted by infiltration
        :return:
        """
        # Starting at 0 since python is 0-based
        wf_that_supplies_free_drainage_demand = self.top_layer.wetting_fronts[0]
        return self.top_layer.calc_wetting_front_free_drainage(
            wf_that_supplies_free_drainage_demand.psi_cm,
            wf_that_supplies_free_drainage_demand,
        )

    def move_wetting_front(
        self,
        infiltration,
        AET_sub,
        ending_volume_sub,
        subtimestep_h,
        bottom_boundary_flux,
    ):
        self.num_wetting_fronts = self.calc_num_wetting_fronts()
        infiltration = self.bottom_layer.move_wetting_fronts(
            infiltration,
            AET_sub,
            ending_volume_sub,
            self.num_wetting_fronts,
            subtimestep_h,
        )
        self.top_layer.merge_wetting_fronts()
        self.top_layer.wetting_fronts_cross_layer_boundary()
        self.top_layer.merge_wetting_fronts()
        bottom_boundary_flux = (
            bottom_boundary_flux + self.top_layer.wetting_front_cross_domain_boundary()
        )
        infiltration = bottom_boundary_flux
        mass_change = self.top_layer.fix_dry_over_wet_fronts()
        if torch.abs(mass_change) > 1e-7:
            AET_sub = AET_sub - mass_change
        self.top_layer.update_psi()
        return infiltration, AET_sub

    def update_ponded_depth(self, ponded_depth_sub):
        if ponded_depth_sub < self.global_params.ponded_depth_max:
            runoff_sub = torch.tensor(0.0, device=self.cfg.device)
            self.runoff = self.runoff + runoff_sub
            ponded_water_sub = ponded_depth_sub
            ponded_depth_sub = torch.tensor(0.0, device=self.cfg.device)

        else:
            # This is the timestep that adds runoff
            runoff_sub = ponded_depth_sub - self.global_params.ponded_depth_max
            self.runoff = self.runoff + runoff_sub
            ponded_water_sub = ponded_depth_sub
            ponded_depth_sub = self.global_params.ponded_depth_max
        return ponded_depth_sub, ponded_water_sub, runoff_sub

    def print_params(self):
        for i in range(len(self.alpha)):
            alpha = self.alpha[i]
            log.info(f"Alpha for soil {i+1}: {alpha.detach().item():.4f}")
        for i in range(len(self.n)):
            n = self.n[i]
            log.info(f"n for soil {i+1}: {n.detach().item():.4f}")
        for i in range(len(self.ksat)):
            ksat = self.ksat[i]
            log.info(f"Ksat for soil {i+1}: {ksat.detach().item():.4f}")
        log.info(f"Max Ponded Depth: {self.ponded_depth_max.detach().item():.4f}")

    def print_local_mass_balance(
        self,
        local_mb,
        starting_volume_sub,
        precip_sub,
        runoff_sub,
        AET_sub,
        ponded_water_sub,
        percolation_sub,
        ending_volume_sub,
        infiltration_sub,
        i,
    ):
        self.top_layer.print()
        log.info(f"Local mass balance at timestep {i}:")
        log.info(f"Error         = {local_mb.item():14.10f}")
        log.info(f"Initial water = {starting_volume_sub.item():14.10f}")
        log.info(f"Water added   = {precip_sub.item():14.10f}")
        log.info(f"Ponded water  = {ponded_water_sub.item():14.10f}")
        log.info(f"Infiltration  = {infiltration_sub.item():14.10f}")
        log.info(f"Runoff        = {runoff_sub.item():14.10f}")
        log.info(f"AET           = {AET_sub.item():14.10f}")
        log.info(f"Percolation   = {percolation_sub.item():14.10f}")
        log.info(f"Final water   = {ending_volume_sub.item():14.10f}")

        # self.print_local_mass_balance(
        #     local_mb,
        #     starting_volume_sub,
        #     precip_sub,
        #     runoff_sub,
        #     AET_sub,
        #     ponded_water_sub,
        #     percolation_sub,
        #     ending_volume_sub,
        #     infiltration_sub,
        # )
