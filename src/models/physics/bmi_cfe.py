import time
import numpy as np
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
from models.physics.cfe import CFE
import torch
from torch import Tensor
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class BMI_CFE:
    def __init__(
        self, refkdt: Tensor, satdk: Tensor, cfg=None, cfe_params=None, verbose=False
    ):
        # ________________________________________________
        # Create a Bmi CFE model that is ready for initialization

        super(BMI_CFE, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max

        # these need to be initialized here as scale_output() called in update()
        self.streamflow_cmh = torch.tensor(0.0)
        # self.streamflow_fms = 0.0
        self.surface_runoff_m = torch.tensor(0.0)

        # ________________________________________________
        # Required, static attributes of the model

        self._att_map = {
            "model_name": "Conceptual Functional Equivalent (CFE)",
            "version": "1.0",
            "author_name": "Jonathan Martin Frame",
            "grid_type": "scalar",
            "time_step_size": 3600,
            "time_units": "1 hour",
        }

        # ________________________________________________
        # Input variable names (CSDMS standard names)

        self._input_var_names = [
            "atmosphere_water__time_integral_of_precipitation_mass_flux",
            "water_potential_evaporation_flux",
        ]

        # ________________________________________________
        # Output variable names (CSDMS standard names)

        self._output_var_names = [
            "land_surface_water__runoff_depth",
            "land_surface_water__runoff_volume_flux",
            "DIRECT_RUNOFF",
            "GIUH_RUNOFF",
            "NASH_LATERAL_RUNOFF",
            "DEEP_GW_TO_CHANNEL_FLUX",
            "SOIL_CONCEPTUAL_STORAGE",
        ]

        # ________________________________________________
        # Create a Python dictionary that maps CSDMS Standard
        # Names to the model's internal variable names.
        # This is going to get long,
        #     since the input variable names could come from any forcing...

        self._var_name_units_map = {
            "land_surface_water__runoff_volume_flux": ["streamflow_cmh", "m3 h-1"],
            "land_surface_water__runoff_depth": ["total_discharge", "m h-1"],
            # --------------   Dynamic inputs --------------------------------
            "atmosphere_water__time_integral_of_precipitation_mass_flux": [
                "timestep_rainfall_input_m",
                "m h-1",
            ],
            "water_potential_evaporation_flux": ["potential_et_m_per_s", "m s-1"],
            "DIRECT_RUNOFF": ["surface_runoff_depth_m", "m"],
            "GIUH_RUNOFF": ["flux_giuh_runoff_m", "m"],
            "NASH_LATERAL_RUNOFF": ["flux_nash_lateral_runoff_m", "m"],
            "DEEP_GW_TO_CHANNEL_FLUX": ["flux_from_deep_gw_to_chan_m", "m"],
            "SOIL_CONCEPTUAL_STORAGE": ["soil_reservoir['storage_m']", "m"],
        }

        # ________________________________________________
        # this is the bmi configuration file
        self.cfe_params = cfe_params
        self.cfg = cfg

        # NN params
        self.refkdt = refkdt
        self.satdk = satdk

        # This takes in the cfg read with Hydra from the yml file
        # self.cfe_cfg = global_params

        self.num_basins = len(self.cfg.data.basin_ids)

        # Verbose
        self.verbose = verbose

        # _________________________________________________
        # nn parameters
        # None

    def load_cfe_params(self):
        # GET VALUES FROM Data class.

        # Catchment area
        self.catchment_area_km2 = self.cfe_params["catchment_area_km2"]

        # Soil parameters
        self.alpha_fc = self.cfe_params["alpha_fc"]
        self.soil_params = self.cfe_params["soil_params"]
        self.soil_params["scheme"] = self.cfg.soil_scheme

        # GW paramters
        self.max_gw_storage = self.cfe_params["max_gw_storage"]
        self.expon = self.cfe_params["expon"]
        self.Cgw = self.cfe_params["Cgw"]

        # Nash storage
        self.K_nash = self.cfe_params["K_nash"]
        self.nash_storage = self.cfe_params["nash_storage"].view(self.num_basins, -1)

        # Lateral flow
        self.K_lf = self.cfe_params["K_lf"]

        # Surface runoff
        self.giuh_ordinates = self.cfe_params["giuh_ordinates"].view(
            self.num_basins, -1
        )
        self.surface_partitioning_scheme = self.cfe_params["partition_scheme"]

        # Other
        self.stand_alone = 0

    # __________________________________________________________________
    # __________________________________________________________________
    # BMI: Model Control Function
    def initialize(self, current_time_step=0):
        self.current_time_step = current_time_step

        # ________________________________________________
        # Create some lookup tabels from the long variable names
        self._var_name_map_long_first = {
            long_name: self._var_name_units_map[long_name][0]
            for long_name in self._var_name_units_map.keys()
        }
        self._var_name_map_short_first = {
            self._var_name_units_map[long_name][0]: long_name
            for long_name in self._var_name_units_map.keys()
        }
        self._var_units_map = {
            long_name: self._var_name_units_map[long_name][1]
            for long_name in self._var_name_units_map.keys()
        }

        # ________________________________________________
        # Initalize all the variables
        # so that they'll be picked up with the get functions
        for long_var_name in list(self._var_name_units_map.keys()):
            # All the variables are single values
            # so just set to zero for now
            self._values[long_var_name] = 0
            setattr(self, self.get_var_name(long_var_name), 0)

        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.load_cfe_params()

        # ________________________________________________
        # initialize simulation constants
        self.atm_press_Pa = 101325.0
        self.unit_weight_water_N_per_m3 = 9810.0

        # ________________________________________________
        # Time control
        self.time_step_size = 3600
        self.timestep_h = self.time_step_size / 3600
        self.timestep_d = self.timestep_h / 24.0

        # ________________________________________________________
        # Set these values now that we have the information from the configuration file.
        self.num_giuh_ordinates = self.giuh_ordinates.size(1)
        self.num_lateral_flow_nash_reservoirs = self.nash_storage.size(1)
        # ________________________________________________
        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        self.output_factor_cms = (
            (1 / 1000) * (self.catchment_area_km2 * 1000 * 1000) * (1 / 3600)
        )

        # ________________________________________________
        # The configuration should let the BMI know what mode to run in (framework vs standalone)
        # If it is stand alone, then load in the forcing and read the time from the forcig file
        if self.stand_alone == 1:
            self.load_forcing_file()
            try:
                self.current_time = pd.to_datetime(
                    self.forcing_data["time"][self.current_time_step]
                )
            except:
                try:
                    self.current_time = pd.to_datetime(
                        self.forcing_data["date"][self.current_time_step]
                    )
                except:
                    print("Check the column names")
        # ________________________________________________
        # In order to check mass conservation at any time
        self.reset_volume_tracking()
        self.reset_flux_and_states()

        ####################################################################
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        # CREATE AN INSTANCE OF THE CONCEPTUAL FUNCTIONAL EQUIVALENT MODEL #
        self.cfe_model = CFE()
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        ####################################################################

    # ________________________________________________
    # Reset the flux and states to zero for the next epoch in NN
    def reset_flux_and_states(self):
        # ________________________________________________
        # Time control
        self.current_time_step = 0
        self.current_time = pd.Timestamp(year=2007, month=10, day=1, hour=0)

        # ________________________________________________
        # Inputs
        self.timestep_rainfall_input_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.potential_et_m_per_s = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________
        # flux variables
        self.flux_overland_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # surface runoff that goes through the GIUH convolution process
        self.flux_perc_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # flux from soil to deeper groundwater reservoir
        self.flux_lat_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # lateral flux in the subsurface to the Nash cascade
        self.flux_from_deep_gw_to_chan_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # flux from the deep reservoir into the channels
        self.gw_reservoir_storage_deficit_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # the available space in the conceptual groundwater reservoir
        self.primary_flux = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # temporary vars.
        self.secondary_flux = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )  # temporary vars.
        self.primary_flux_from_gw_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.secondary_flux_from_gw_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.total_discharge = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.diff_infilt = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.diff_perc = torch.zeros((1, self.num_basins), dtype=torch.float64)

        # ________________________________________________
        # Evapotranspiration
        self.potential_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.reduced_potential_et_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_from_rain_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.actual_et_from_soil_m_per_timestep = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________
        # ________________________________________________
        # SOIL RESERVOIR CONFIGURATION
        # Local values to be used in setting up soil reservoir
        trigger_z_m = torch.tensor([0.5])
        field_capacity_atm_press_fraction = self.alpha_fc

        # ________________________________________________
        # Soil outflux calculation, Equation 3 in Fred Ogden's document

        H_water_table_m = (
            field_capacity_atm_press_fraction
            * self.atm_press_Pa
            / self.unit_weight_water_N_per_m3
        )

        soil_water_content_at_field_capacity = self.soil_params["smcmax"] * torch.pow(
            H_water_table_m / self.soil_params["satpsi"], (1.0 / self.soil_params["bb"])
        )

        Omega = H_water_table_m - trigger_z_m

        # ________________________________________________
        # Upper & lower limit of the integral in Equation 4 in Fred Ogden's document

        lower_lim = torch.pow(Omega, (1.0 - 1.0 / self.soil_params["bb"])) / (
            1.0 - 1.0 / self.soil_params["bb"]
        )

        upper_lim = torch.pow(
            Omega + self.soil_params["D"], (1.0 - 1.0 / self.soil_params["bb"])
        ) / (1.0 - 1.0 / self.soil_params["bb"])

        # ________________________________________________
        # Integral & power term in Equation 4 & 5 in Fred Ogden's document

        storage_thresh_pow_term = torch.pow(
            1.0 / self.soil_params["satpsi"], (-1.0 / self.soil_params["bb"])
        )

        lim_diff = upper_lim - lower_lim

        field_capacity_storage_threshold_m = (
            self.soil_params["smcmax"] * storage_thresh_pow_term * lim_diff
        )

        # ________________________________________________
        # lateral flow function parameters
        assumed_near_channel_water_table_slope = 0.01  # [L/L]
        lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m
        self.soil_reservoir_storage_deficit_m = torch.tensor([0.0], dtype=torch.float64)

        # ________________________________________________
        # Subsurface reservoirs
        self.gw_reservoir = {
            "is_exponential": True,
            "storage_max_m": self.max_gw_storage,
            "coeff_primary": self.Cgw,
            "exponent_primary": self.expon,
            "storage_threshold_primary_m": torch.zeros(
                (1, self.num_basins), dtype=torch.float64
            ),
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            "storage_threshold_secondary_m": torch.zeros(
                (1, self.num_basins), dtype=torch.float64
            ),
            "coeff_secondary": torch.zeros((1, self.num_basins), dtype=torch.float64),
            "exponent_secondary": torch.ones((1, self.num_basins), dtype=torch.float64),
        }
        self.gw_reservoir["storage_m"] = self.gw_reservoir["storage_max_m"] * 0.01
        self.volstart = self.volstart.add(self.gw_reservoir["storage_m"])
        self.vol_in_gw_start = self.gw_reservoir["storage_m"]

        self.soil_reservoir = {
            "is_exponential": False,
            "wilting_point_m": self.soil_params["wltsmc"] * self.soil_params["D"],
            "storage_max_m": self.soil_params["smcmax"] * self.soil_params["D"],
            "coeff_primary": self.satdk
            * self.soil_params["slop"]
            * self.time_step_size,  # Controls percolation to GW, Equation 11
            "exponent_primary": torch.ones(
                (1, self.num_basins), dtype=torch.float64
            ),  # Controls percolation to GW, FIXED to 1 based on Equation 11
            "storage_threshold_primary_m": field_capacity_storage_threshold_m,
            "coeff_secondary": self.K_lf,  # Controls lateral flow
            "exponent_secondary": torch.ones(
                (1, self.num_basins), dtype=torch.float64
            ),  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            "storage_threshold_secondary_m": lateral_flow_threshold_storage_m,
        }
        self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_max_m"] * 0.6
        self.volstart = self.volstart.add(self.soil_reservoir["storage_m"])
        self.vol_soil_start = self.soil_reservoir["storage_m"]

        # ________________________________________________
        # Schaake partitioning
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.refkdt * self.satdk / 2.0e-06
        )
        self.Schaake_output_runoff_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.infiltration_depth_m = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )

        # ________________________________________________________
        self.runoff_queue_m_per_timestep = torch.zeros(
            self.giuh_ordinates.shape[0], self.num_giuh_ordinates + 1
        )

    def update_params(self, refkdt, satdk):
        """Update dynamic parameters"""
        self.refkdt = refkdt
        self.satdk = satdk
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.refkdt * self.satdk / 2.0e-06
        )
        self.soil_reservoir["coeff_primary"] = self.satdk
        if self.verbose:
            print(
                f"refkdt: {self.refkdt:.2f}; satdk: {self.satdk:.5f}; \
                Schaake: {self.Schaake_adjusted_magic_constant_by_soil_type:.3f};\
                Soilcoeff: {self.soil_reservoir['coeff_primary']:.5f}"
            )

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update(self):
        self.volin = self.volin.add(self.timestep_rainfall_input_m)
        self.cfe_model.run_cfe(self)
        self.scale_output()

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update_until(self, until, verbose=True):
        for i in range(self.current_time_step, until):
            self.cfe_model.run_cfe(self)
            self.scale_output()
            if verbose:
                print("total discharge: {}".format(self.total_discharge))
                print("at time: {}".format(self.current_time))

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def finalize(self, print_mass_balance=False):
        self.finalize_mass_balance(verbose=print_mass_balance)
        self.reset_volume_tracking()

        """Finalize model."""
        self.cfe_model = None
        self.cfe_state = None

    # ________________________________________________
    # Mass balance tracking
    def reset_volume_tracking(self):
        self.volstart = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_rain = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_partition_runoff = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_partition_infilt = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_out_giuh = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_end_giuh = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw_start = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_gw_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_from_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_in_nash = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_in_nash_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_out_nash = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_start = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_to_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_to_lat_flow = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_soil_to_gw = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_soil_end = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volin = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volout = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.volend = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_partition_runoff_IOF = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_partition_runoff_SOF = torch.zeros(
            (1, self.num_basins), dtype=torch.float64
        )
        self.vol_et_to_atm = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_soil = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_et_from_rain = torch.zeros((1, self.num_basins), dtype=torch.float64)
        self.vol_PET = torch.zeros((1, self.num_basins), dtype=torch.float64)
        return

    # ________________________________________________________
    def finalize_mass_balance(self, verbose=True):
        self.volend = self.soil_reservoir["storage_m"] + self.gw_reservoir["storage_m"]
        self.vol_in_gw_end = self.gw_reservoir["storage_m"]

        # the GIUH queue might have water in it at the end of the simulation, so sum it up.
        self.vol_end_giuh = torch.sum(self.runoff_queue_m_per_timestep, dim=1)
        self.vol_in_nash_end = torch.sum(self.nash_storage, dim=1)

        self.vol_soil_end = self.soil_reservoir["storage_m"]

        self.global_residual = (
            self.volstart + self.volin - self.volout - self.volend - self.vol_end_giuh
        )
        self.partition_residual = (
            self.volin
            - self.vol_partition_runoff
            - self.vol_partition_infilt
            - self.vol_et_from_rain
        )
        self.giuh_residual = (
            self.vol_partition_runoff - self.vol_out_giuh - self.vol_end_giuh
        )
        self.soil_residual = (
            self.vol_soil_start
            + self.vol_to_soil
            - self.vol_soil_to_lat_flow
            - self.vol_to_gw
            - self.vol_et_from_soil
            - self.vol_soil_end
        )
        self.nash_residual = self.vol_in_nash - self.vol_out_nash - self.vol_in_nash_end
        self.gw_residual = (
            self.vol_in_gw_start
            + self.vol_to_gw
            - self.vol_from_gw
            - self.vol_in_gw_end
        )
        self.AET_residual = (
            self.vol_et_to_atm - self.vol_et_from_rain - self.vol_et_from_soil
        )

        if verbose:
            i = 0
            print(f"\nGLOBAL MASS BALANCE (print for {i}-th basin)")
            print("  initial volume: {:8.4f}".format(self.volstart[0][i].item()))
            print("    volume input: {:8.4f}".format(self.volin[0][i].item()))
            print("   volume output: {:8.4f}".format(self.volout[0][i].item()))
            print("    final volume: {:8.4f}".format(self.volend[0][i].item()))
            print("        residual: {:6.4e}".format(self.global_residual[0][i].item()))

            print("\n AET & PET")
            print("      volume PET: {:8.4f}".format(self.vol_PET[0][i].item()))
            print("      volume AET: {:8.4f}".format(self.vol_et_to_atm[0][i].item()))
            print(
                "ET from rainfall: {:8.4f}".format(self.vol_et_from_rain[0][i].item())
            )
            print(
                "    ET from soil: {:8.4f}".format(self.vol_et_from_soil[0][i].item())
            )
            print("    AET residual: {:6.4e}".format(self.AET_residual[0][i].item()))

            print("\nPARTITION MASS BALANCE")
            print(
                "    surface runoff: {:8.4f}".format(
                    self.vol_partition_runoff[0][i].item()
                )
            )
            print(
                "      infiltration: {:8.4f}".format(
                    self.vol_partition_infilt[0][i].item()
                )
            )
            print(
                " vol. et from rain: {:8.4f}".format(self.vol_et_from_rain[0][i].item())
            )
            print(
                "partition residual: {:6.4e}".format(
                    self.partition_residual[0][i].item()
                )
            )

            print("\nGIUH MASS BALANCE")
            print(
                "  vol. into giuh: {:8.4f}".format(
                    self.vol_partition_runoff[0][i].item()
                )
            )
            print("   vol. out giuh: {:8.4f}".format(self.vol_out_giuh[0][i].item()))
            print(" vol. end giuh q: {:8.4f}".format(self.vol_end_giuh[i].item()))
            print("   giuh residual: {:6.4e}".format(self.giuh_residual[0][i].item()))

            if self.soil_params["scheme"] == "classic":
                print("\nSOIL WATER CONCEPTUAL RESERVOIR MASS BALANCE")
            elif self.soil_params["scheme"] == "ode":
                print("\nSOIL WATER MASS BALANCE")
            print(
                "     init soil vol: {:8.6f}".format(self.vol_soil_start[0][i].item())
            )
            print("    vol. into soil: {:8.6f}".format(self.vol_to_soil[0][i].item()))
            print(
                "  vol.soil2latflow: {:8.6f}".format(
                    self.vol_soil_to_lat_flow[0][i].item()
                )
            )
            print(
                "   vol. soil to gw: {:8.6f}".format(self.vol_soil_to_gw[0][i].item())
            )
            print(
                " vol. et from soil: {:8.6f}".format(self.vol_et_from_soil[0][i].item())
            )
            print("   final vol. soil: {:8.6f}".format(self.vol_soil_end[0][i].item()))
            print("  vol. soil resid.: {:6.6e}".format(self.soil_residual[0][i].item()))

            print("\nNASH CASCADE CONCEPTUAL RESERVOIR MASS BALANCE")
            print("    vol. to nash: {:8.4f}".format(self.vol_in_nash[0][i].item()))
            print("  vol. from nash: {:8.4f}".format(self.vol_out_nash[0][i].item()))
            print(" final vol. nash: {:8.4f}".format(self.vol_in_nash_end[i].item()))
            print("nash casc resid.: {:6.4e}".format(self.nash_residual[0][i].item()))

            print("\nGROUNDWATER CONCEPTUAL RESERVOIR MASS BALANCE")
            print("init gw. storage: {:8.4f}".format(self.vol_in_gw_start[0][i].item()))
            print("       vol to gw: {:8.4f}".format(self.vol_to_gw[0][i].item()))
            print("     vol from gw: {:8.4f}".format(self.vol_from_gw[0][i].item()))
            print("final gw.storage: {:8.4f}".format(self.vol_in_gw_end[0][i].item()))
            print("    gw. residual: {:6.4e}".format(self.gw_residual[0][i].item()))

        return

    # ________________________________________________________
    def load_forcing_file(self):
        self.forcing_data = pd.read_csv(self.forcing_file)

    # ________________________________________________________
    def load_unit_test_data(self):
        self.unit_test_data = pd.read_csv(self.compare_results_file)
        self.cfe_output_data = pd.DataFrame().reindex_like(self.unit_test_data)

    # ------------------------------------------------------------
    def scale_output(self):
        self.surface_runoff_m = self.flux_Qout_m  # self.total_discharge
        self._values["land_surface_water__runoff_depth"] = self.surface_runoff_m
        self.streamflow_cmh = (
            self.total_discharge
        )  # self._values['land_surface_water__runoff_depth'] * self.output_factor_cms

        self._values[
            "land_surface_water__runoff_volume_flux"
        ] = self.streamflow_cmh  # * (1/35.314)

        self._values["DIRECT_RUNOFF"] = self.surface_runoff_depth_m
        self._values["GIUH_RUNOFF"] = self.flux_giuh_runoff_m
        self._values["NASH_LATERAL_RUNOFF"] = self.flux_nash_lateral_runoff_m
        self._values["DEEP_GW_TO_CHANNEL_FLUX"] = self.flux_from_deep_gw_to_chan_m
        # if self.soil_scheme.lower() == 'ode': # Commented out just for debugging, restore later
        self._values["SOIL_CONCEPTUAL_STORAGE"] = self.soil_reservoir["storage_m"]

    # ----------------------------------------------------------------------------
    def initialize_forcings(self):
        for forcing_name in self.cfg_train["dynamic_inputs"]:
            setattr(self, self._var_name_map_short_first[forcing_name], 0)

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Model Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    def get_attribute(self, att_name):
        try:
            return self._att_map[att_name.lower()]
        except:
            print(" ERROR: Could not find attribute: " + att_name)

    # --------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    # --------------------------------------------------------
    def get_input_var_names(self):
        return self._input_var_names

    def get_output_var_names(self):
        return self._output_var_names

    # ------------------------------------------------------------
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute("model_name")  # JG Edit

    # ------------------------------------------------------------
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    # ------------------------------------------------------------
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    # ------------------------------------------------------------
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name)

    def return_runoff(self):
        return self.flux_Qout_m

    # -------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Variable Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    def get_var_name(self, long_var_name):
        return self._var_name_map_long_first[long_var_name]

    # -------------------------------------------------------------------
    def get_var_units(self, long_var_name):
        return self._var_units_map[long_var_name]

    # -------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        # JG Edit
        return self.get_value_ptr(long_var_name)  # .dtype

    # ------------------------------------------------------------
    def get_var_grid(self, name):
        # JG Edit
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id

    # ------------------------------------------------------------
    def get_var_itemsize(self, name):
        #        return np.dtype(self.get_var_type(name)).itemsize
        return np.array(self.get_value(name)).itemsize

    # ------------------------------------------------------------
    def get_var_location(self, name):
        # JG Edit
        # all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    # -------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):
        return np.int16(0)

    # -------------------------------------------------------------------
    def get_start_time(self):
        return self._start_time  # JG Edit

    # -------------------------------------------------------------------
    def get_end_time(self):
        return self._end_time  # JG Edit

    # -------------------------------------------------------------------
    def get_current_time(self):
        return self.current_time

    # -------------------------------------------------------------------
    def get_time_step(self):
        return self.get_attribute("time_step_size")  # JG: Edit

    # -------------------------------------------------------------------
    def get_time_units(self):
        return self.get_attribute("time_units")

    # -------------------------------------------------------------------
    def set_value(self, var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """
        setattr(self, self.get_var_name(var_name), value)
        self._values[var_name] = value

    # ------------------------------------------------------------
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        # JG Note: TODO confirm this is correct. Get/set values ~=
        #        val = self.get_value_ptr(name)
        #        val.flat[inds] = src

        # JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(name)).flatten().shape[0] == 1:
            self.set_value(name, src)
        else:
            # JMFrame: Need to set the value with the updated array with new index value
            val = self.get_value_ptr(name)
            for i in inds.shape:
                val.flatten()[inds[i]] = src[i]
            self.set_value(name, val)

    # ------------------------------------------------------------
    def get_var_nbytes(self, long_var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        # JMFrame NOTE: Had to import sys for this function
        return sys.getsizeof(self.get_value_ptr(long_var_name))

    # ------------------------------------------------------------
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        # JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(var_name)).flatten().shape[0] == 1:
            return self.get_value(var_name)
        else:
            val_array = self.get_value(var_name).flatten()
            return np.array([val_array[i] for i in indices])

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html
    # ------------------------------------------------------------
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    # ------------------------------------------------------------
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    # ------------------------------------------------------------
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    # ------------------------------------------------------------
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    # ------------------------------------------------------------
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    # ------------------------------------------------------------
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    # ------------------------------------------------------------
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    # ------------------------------------------------------------
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin")

    # ------------------------------------------------------------
    def get_grid_rank(self, grid_id):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    # ------------------------------------------------------------
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape")

    # ------------------------------------------------------------
    def get_grid_size(self, grid_id):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    # ------------------------------------------------------------
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing")

    # ------------------------------------------------------------
    def get_grid_type(self, grid_id=0):
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return "scalar"

    # ------------------------------------------------------------
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x")

    # ------------------------------------------------------------
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y")

    # ------------------------------------------------------------
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z")
