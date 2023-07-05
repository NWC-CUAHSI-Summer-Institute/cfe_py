import time
import numpy as np
import pandas as pd
import sys
# from scipy.integrate import odeint
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

class soil_moisture_flux_ode(nn.Module):
    """
    Soil reservoir module that solves ODE
    Using ODE allows simultaneous calculation of outflux, instead of stepwise subtraction of flux which causes overextraction from SM reservoir
    The behavior of soil moisture storage is divided into 3 stages. 
    Stage 1: S (Soil moisture storage ) > storage_threshold_primary_m
        Interpretation: When the soil moisture is plenty, AET(=PET), percolation, and lateral flow are all active.
        Equation: dS/dt = Infiltration - PET - (Klf+Kperc) * (S - storage_threshold_primary_m)/(storage_max_m - storage_threshold_primary_m)
    Stage 2: storage_threshold_primary_m > S (Soil moisture storage) > storage_threshold_primary_m - wltsmc
        Interpretation: When the soil moisture is in the medium range, AET is active and proportional to the soil moisture storage ratio. No percolation and lateral flow fluxes. 
        Equation: dS/dt = Infiltration - PET * (S - wltsmc)/(storage_threshold_primary_m - wltsmc)
    Stage 3: wltsmc > S (Soil moisture storage)
        Interpretation: When the soil moisture is depleted, no outflux is active
        Equation: dS/dt = Infitlation
        
    :param t: time
    :param S: Soil moisture storage in meter
    :param storage_threshold_primary_m:
    :param storage_max_m: maximum soil moisture storage, i.e., porosity
    :param coeff_primary: K_perc, percolation coefficient
    :param coeff_secondary: K_lf, lateral flow coefficient
    :param PET: potential evapotranspiration
    :param infilt: infiltration
    :param wilting_point_m: wilting point (in meter)
    :return: dS
    """
    
    def __init__(self, cfe_state=None, reservoir=None):
        super().__init__()
        self.cfe_state = cfe_state
        self.reservoir = reservoir

    def forward(self, t, states):
        
        S = states
            
        storage_above_threshold_m = S - self.reservoir['storage_threshold_primary_m']
        storage_diff = self.reservoir['storage_max_m'] - self.reservoir['storage_threshold_primary_m']
        storage_ratio = torch.minimum(storage_above_threshold_m / storage_diff, torch.tensor([1.0]))

        perc_lat_switch = torch.multiply(S - self.reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = torch.multiply(S - self.reservoir['wilting_point_m'] > 0, 1)

        storage_above_threshold_m_paw = S - self.reservoir['wilting_point_m']
        storage_diff_paw = self.reservoir['storage_threshold_primary_m'] - self.reservoir['wilting_point_m']
        storage_ratio_paw = torch.minimum(storage_above_threshold_m_paw / storage_diff_paw, torch.tensor([0.3])) # Equation 11 (Ogden's document)
        dS_dt = self.cfe_state.infiltration_depth_m -1 * perc_lat_switch * (self.reservoir['coeff_primary'] + self.reservoir['coeff_secondary']) * storage_ratio - ET_switch * self.cfe_state.reduced_potential_et_m_per_timestep * storage_ratio_paw
        
        return (dS_dt)
    

class CFE():
    def __init__(self):
        super(CFE, self).__init__()
        
    # ____________________________________________________________________________________
    def calculate_input_rainfall_and_PET(self, cfe_state):
        """
        Calculate input rainfall and PET 
        """
        cfe_state.potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        
    # ____________________________________________________________________________________
    def calculate_evaporation_from_rainfall(self, cfe_state):
        """
        Calculate evaporation from rainfall. If it is raining, take PET from rainfall
        """
        cfe_state.actual_et_from_rain_m_per_timestep = torch.tensor(0.0, dtype=torch.float)
        if(cfe_state.timestep_rainfall_input_m > 0):
            self.et_from_rainfall(cfe_state)
        
        cfe_state.vol_et_from_rain = cfe_state.vol_et_from_rain.add(cfe_state.actual_et_from_rain_m_per_timestep)
        cfe_state.vol_et_to_atm = cfe_state.vol_et_to_atm.add(cfe_state.actual_et_from_rain_m_per_timestep)
        cfe_state.volout = cfe_state.volout.add(cfe_state.actual_et_from_rain_m_per_timestep)
        
        cfe_state.actual_et_m_per_timestep = cfe_state.actual_et_m_per_timestep.add(cfe_state.actual_et_from_rain_m_per_timestep)
        
    # ____________________________________________________________________________________
    def calculate_evaporation_from_soil(self, cfe_state):
        """
        If the soil moisture calculation scheme is 'classic', calculate the evaporation from the soil
        Elseif the soil moisture calculation scheme is 'ode', do nothing, because evaporation from the soil will be calculated within run_soil_moisture_scheme
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            
            cfe_state.actual_et_from_soil_m_per_timestep = torch.tensor(0.0, dtype=torch.float)
            # If the soil moisture storage is more than wilting point, calculate ET from soil
            if(cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']): 
                self.et_from_soil(cfe_state)

            cfe_state.vol_et_from_soil = cfe_state.vol_et_from_soil.add(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.vol_et_to_atm = cfe_state.vol_et_to_atm.add(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.volout = cfe_state.volout.add(cfe_state.actual_et_from_soil_m_per_timestep)

            cfe_state.actual_et_m_per_timestep = cfe_state.actual_et_m_per_timestep.add(cfe_state.actual_et_from_soil_m_per_timestep)
            
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            None
        
    # ____________________________________________________________________________________
    def calculate_the_soil_moisture_deficit(self, cfe_state):
        """ Calculate the soil moisture deficit
        """
        cfe_state.soil_reservoir_storage_deficit_m = (cfe_state.soil_params['smcmax'] * cfe_state.soil_params['D'] - \
                                                        cfe_state.soil_reservoir['storage_m'])
        
    # ____________________________________________________________________________________
    def calculate_infiltration_excess_overland_flow(self, cfe_state):
        """ Calculates infiltration excess overland flow 
        by running the partitioning scheme based on the choice set in the Configuration file
        """
        if (cfe_state.timestep_rainfall_input_m > 0.0): 
            if cfe_state.surface_partitioning_scheme == "Schaake": 
                self.Schaake_partitioning_scheme(cfe_state)
            elif cfe_state.surface_partitioning_scheme == "Xinanjiang": 
                self.Xinanjiang_partitioning_scheme(cfe_state)
            else: 
                print("Problem: must specify one of Schaake of Xinanjiang partitioning scheme.\n")
                print("Program terminating.:( \n");
                sys.exit(1)
        else: 
            cfe_state.surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float)
            cfe_state.infiltration_depth_m = torch.tensor(0.0, dtype=torch.float)

    # __________________________________________________________________________________________________________
    def calculate_saturation_excess_overland_flow_from_soil(self, cfe_state):
        """ Calculates saturation excess overland flow (SOF)
        This should be run after calculate_infiltration_excess_overland_flow, then, 
        infiltration_depth_m and surface_runoff_depth_m get finalized 
        """
        # If the infiltration is more than the soil moisture deficit, 
        # additional runoff (SOF) occurs and soil get saturated
        if cfe_state.soil_reservoir_storage_deficit_m < cfe_state.infiltration_depth_m:
            diff = cfe_state.infiltration_depth_m - cfe_state.soil_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m = cfe_state.surface_runoff_depth_m.add(diff)
            cfe_state.infiltration_depth_m = cfe_state.infiltration_depth_m.sub(diff)
            cfe_state.soil_reservoir_storage_deficit_m = torch.tensor(0.0, dtype=torch.float)
        else:
            # If the infiltration is less than the soil moisture deficit,
            # Infiltration & runoff flux is as calculated in calculate_infiltration_excess_overland_flow()
            None

    # __________________________________________________________________________________________________________
    def track_infiltration_and_runoff(self, cfe_state):
        """ Tracking runoff & infiltraiton volume with final infiltration & runoff values
        """
        cfe_state.vol_partition_runoff = cfe_state.vol_partition_runoff.add(cfe_state.surface_runoff_depth_m)
        cfe_state.vol_partition_infilt = cfe_state.vol_partition_infilt.add(cfe_state.infiltration_depth_m)
        cfe_state.vol_to_soil = cfe_state.vol_to_soil.add(cfe_state.infiltration_depth_m)
        
    # __________________________________________________________________________________________________________        
    def run_soil_moisture_scheme(self, cfe_state):
        """ Run the soil moisture scheme based on the choice set in the Configuration file
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            # Add infiltration flux and calculate the reservoir flux 
            cfe_state.soil_reservoir['storage_m'] = cfe_state.soil_reservoir['storage_m'].add(cfe_state.infiltration_depth_m)
            self.soil_conceptual_reservoir_flux_calc(cfe_state, cfe_state.soil_reservoir)
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            # Infiltration flux is added witin the ODE scheme
            self.soil_moisture_flux_calc_with_ode(cfe_state=cfe_state, reservoir=cfe_state.soil_reservoir)

    # ________________________________________________________________________________________________________
    def update_outflux_from_soil(self, cfe_state):
        cfe_state.flux_perc_m = cfe_state.primary_flux_m  # percolation_flux
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m # lateral_flux
        
        # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
        # If ODE, outfluxes are already subtracted from the soil moisture storage
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            cfe_state.soil_reservoir['storage_m'] = cfe_state.soil_reservoir['storage_m'].sub(cfe_state.flux_perc_m)
            cfe_state.soil_reservoir['storage_m'] = cfe_state.soil_reservoir['storage_m'].sub(cfe_state.flux_lat_m)
        
        # If ODE, track actual ET from soil
        if cfe_state.soil_params['scheme'].lower() == 'ode':

            cfe_state.vol_et_from_soil = cfe_state.vol_et_from_soil.add(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.vol_et_to_atm = cfe_state.vol_et_to_atm.add(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.volout = cfe_state.volout.add(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.actual_et_m_per_timestep = cfe_state.actual_et_m_per_timestep.add(cfe_state.actual_et_from_soil_m_per_timestep)

        elif cfe_state.soil_params['scheme'].lower() == 'classic':
            None

    # ________________________________________________________________________________________________________
    def calculate_groundwater_storage_deficit(self, cfe_state):
        cfe_state.gw_reservoir_storage_deficit_m = cfe_state.gw_reservoir['storage_max_m'] - cfe_state.gw_reservoir['storage_m']
        
    # __________________________________________________________________________________________________________
    def calculate_saturation_excess_overland_flow_from_gw(self, cfe_state):
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        if cfe_state.flux_perc_m > cfe_state.gw_reservoir_storage_deficit_m:
            diff = cfe_state.flux_perc_m - cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m = cfe_state.surface_runoff_depth_m.add(diff)
            cfe_state.flux_perc_m = cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.gw_reservoir['storage_m'] = cfe_state.gw_reservoir['storage_max_m']
            cfe_state.gw_reservoir_storage_deficit_m = torch.tensor(0.0, dtype=torch.float)
            
            cfe_state.vol_partition_runoff = cfe_state.vol_partition_runoff.add(diff)
            cfe_state.vol_partition_infilt = cfe_state.vol_partition_infilt.sub(diff)
            
        cfe_state.gw_reservoir['storage_m'] = cfe_state.gw_reservoir['storage_m'].add(cfe_state.flux_perc_m)
            
    # __________________________________________________________________________________________________________
    def track_volume_from_percolation_and_lateral_flow(self, cfe_state):
        # Finalize the percolation and lateral flow 
        cfe_state.vol_to_gw  = cfe_state.vol_to_gw.add(cfe_state.flux_perc_m)
        cfe_state.vol_soil_to_gw = cfe_state.vol_soil_to_gw.add(cfe_state.flux_perc_m)
        cfe_state.vol_soil_to_lat_flow = cfe_state.vol_soil_to_lat_flow.add(cfe_state.flux_lat_m)  #TODO add this to nash cascade as input
        cfe_state.volout = cfe_state.volout.add(cfe_state.flux_lat_m)

    # __________________________________________________________________________________________________________
    
    def set_flux_from_deep_gw_to_chan_m(self, cfe_state):
        
        # Rename the flux from ground wter to deep groundwater to channel 
        cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.primary_flux_from_gw_m
        
        # If the flux from GW storage is larger than the current storage, extract them all
        if (cfe_state.flux_from_deep_gw_to_chan_m >= cfe_state.gw_reservoir['storage_m']): 
            cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.gw_reservoir['storage_m']
            if cfe_state.verbose:
                print("WARNING: Groundwater flux larger than storage. \n")

        # Mass balance
        cfe_state.vol_from_gw = cfe_state.vol_from_gw.add(cfe_state.flux_from_deep_gw_to_chan_m)
        
    # __________________________________________________________________________________________________________
    def remove_flux_from_deep_gw_to_chan_m(self, cfe_state):
        
        # If the flux from GW storage is larger than the current storage, extract them all
        # if (cfe_state.flux_from_deep_gw_to_chan_m >= cfe_state.gw_reservoir['storage_m']):
        #     cfe_state.gw_reservoir['storage_m'] = torch.tensor(0.0, dtype=torch.float)
        # # Else, just extract the amount 
        # else:
        cfe_state.gw_reservoir['storage_m'] = cfe_state.gw_reservoir['storage_m'].sub(cfe_state.flux_from_deep_gw_to_chan_m)
            
    # __________________________________________________________________________________________________________
    def track_volume_from_giuh(self, cfe_state):
        cfe_state.vol_out_giuh = cfe_state.vol_out_giuh.add(cfe_state.flux_giuh_runoff_m)
        cfe_state.volout = cfe_state.volout.add(cfe_state.flux_giuh_runoff_m)
    # __________________________________________________________________________________________________________
    def track_volume_from_deep_gw_to_chan(self, cfe_state):
        cfe_state.volout = cfe_state.volout.add(cfe_state.flux_from_deep_gw_to_chan_m)
    # __________________________________________________________________________________________________________
    def track_volume_from_nash_cascade(self, cfe_state):
        cfe_state.vol_in_nash = cfe_state.vol_in_nash.add(cfe_state.flux_lat_m)
        cfe_state.vol_out_nash = cfe_state.vol_out_nash.add(cfe_state.flux_nash_lateral_runoff_m)
    # __________________________________________________________________________________________________________
    def add_up_total_flux_discharge(self, cfe_state):
        cfe_state.flux_Qout_m = cfe_state.flux_giuh_runoff_m + cfe_state.flux_nash_lateral_runoff_m + cfe_state.flux_from_deep_gw_to_chan_m
        cfe_state.total_discharge = cfe_state.flux_Qout_m * cfe_state.catchment_area_km2 * 1000000.0 / cfe_state.time_step_size
    # __________________________________________________________________________________________________________
    def update_current_time(self, cfe_state):
        cfe_state.current_time_step += 1
        cfe_state.current_time      += pd.Timedelta(value=cfe_state.time_step_size, unit='s')


    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_cfe(self, cfe_state):
        
        # Rainfall and ET 
        if cfe_state.soil_reservoir['storage_m'] < 0:
            print('SM < 0')
            
        # Groundwater reservoir
        if torch.isnan(cfe_state.gw_reservoir['storage_m']):
            print('something with gw is nan')
        if torch.isnan(cfe_state.primary_flux_from_gw_m):
            print('something with gw is nan')
        if torch.isnan(cfe_state.vol_from_gw):
            print('something with gw is nan')
        
        self.calculate_input_rainfall_and_PET(cfe_state)
        self.calculate_evaporation_from_rainfall(cfe_state)
        self.calculate_evaporation_from_soil(cfe_state)
        
        # Infiltration partitioning
        self.calculate_the_soil_moisture_deficit(cfe_state)
        self.calculate_infiltration_excess_overland_flow(cfe_state)
        self.calculate_saturation_excess_overland_flow_from_soil(cfe_state)
        self.track_infiltration_and_runoff(cfe_state)

        # Soil moisture reservoir
        self.run_soil_moisture_scheme(cfe_state)
        self.update_outflux_from_soil(cfe_state)

        # Groundwater reservoir
        self.calculate_groundwater_storage_deficit(cfe_state) # suspicious 
        self.calculate_saturation_excess_overland_flow_from_gw(cfe_state) # This line was fine 
    
        self.track_volume_from_percolation_and_lateral_flow(cfe_state) # Not relevant
        self.gw_conceptual_reservoir_flux_calc(cfe_state=cfe_state, gw_reservoir=cfe_state.gw_reservoir) # No issue 
        
        if not cfe_state.gw_reservoir['storage_m'].requires_grad:
            print('gw_storage is not tracked')
        
        self.set_flux_from_deep_gw_to_chan_m(cfe_state) # suspicious?
    
        # self.check_is_fabs_less_than_epsilon(cfe_state) 
        self.remove_flux_from_deep_gw_to_chan_m(cfe_state) # suspicious? 
        
        # Surface runoff rounting
        self.convolution_integral(cfe_state)
        self.track_volume_from_giuh(cfe_state)
        self.track_volume_from_deep_gw_to_chan(cfe_state)
        
        # Lateral flow rounting
        self.nash_cascade(cfe_state)
        self.track_volume_from_nash_cascade(cfe_state)
        self.add_up_total_flux_discharge(cfe_state)
        
        # Time
        self.update_current_time(cfe_state)
        
        return
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    
    # __________________________________________________________________________________________________________
    def nash_cascade(self, cfe_state):
        """
            Solve for the flow through the Nash cascade to delay the 
            arrival of the lateral flow into the channel
        """
        Q = torch.zeros(cfe_state.num_lateral_flow_nash_reservoirs)

        for i in range(cfe_state.num_lateral_flow_nash_reservoirs):

            Q[i] = cfe_state.K_nash * cfe_state.nash_storage[i]

            # Update nash_storage using `-` and `+` operations instead of .sub() and .add()
            cfe_state.nash_storage[i] = cfe_state.nash_storage[i] - Q[i]

            if i == 0:
                cfe_state.nash_storage[i] = cfe_state.nash_storage[i] + cfe_state.flux_lat_m
            else:
                cfe_state.nash_storage[i] = cfe_state.nash_storage[i] + Q[i-1]

        cfe_state.flux_nash_lateral_runoff_m = Q[cfe_state.num_lateral_flow_nash_reservoirs - 1]

        return
    

    # __________________________________________________________________________________________________________
    # def convolution_integral(self, cfe_state):
    #     """
    #         This function solves the convolution integral involving N GIUH ordinates.
            
    #         Inputs:
    #             Schaake_output_runoff_m
    #             num_giuh_ordinates
    #             giuh_ordinates
    #         Outputs:
    #             runoff_queue_m_per_timestep
    #     """

    #     N = cfe_state.num_giuh_ordinates
    #     updated_runoff_queue = torch.tensor(cfe_state.runoff_queue_m_per_timestep, dtype=torch.float)  # clone to avoid in-place modification
    #     updated_runoff_queue[N] = 0.0

    #     for i in range(N): 
    #         updated_runoff_queue[i] = updated_runoff_queue[i] + (cfe_state.giuh_ordinates[i] * cfe_state.surface_runoff_depth_m)

    #     cfe_state.flux_giuh_runoff_m = updated_runoff_queue[0]

    #     # shift all the entries in preparation for the next timestep
    #     updated_runoff_queue[:-1] = updated_runoff_queue[1:]  # copy values shifted one position left

    #     cfe_state.runoff_queue_m_per_timestep = updated_runoff_queue  # update state with modified tensor

    #     return

    def convolution_integral(self,cfe_state):
        """
            This function solves the convolution integral involving N GIUH ordinates.
            
            Inputs:
                Schaake_output_runoff_m
                num_giuh_ordinates
                giuh_ordinates
            Outputs:
                runoff_queue_m_per_timestep
        """

        N = cfe_state.num_giuh_ordinates

        cfe_state.runoff_queue_m_per_timestep[N] = torch.tensor(0.0, dtype=torch.float)
        
        
        for i in range(cfe_state.num_giuh_ordinates): 

            cfe_state.runoff_queue_m_per_timestep[i] = cfe_state.runoff_queue_m_per_timestep[i].add(cfe_state.giuh_ordinates[i] * cfe_state.surface_runoff_depth_m)
            
        cfe_state.flux_giuh_runoff_m = cfe_state.runoff_queue_m_per_timestep[0]
        
        # __________________________________________________________________
        # shift all the entries in preperation for the next timestep

        for i in range(cfe_state.num_giuh_ordinates): 
            cfe_state.runoff_queue_m_per_timestep[i] = cfe_state.runoff_queue_m_per_timestep[i+1]

        return
    

    # __________________________________________________________________________________________________________
    def et_from_rainfall(self,cfe_state):
        
        """
            iff it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
        """

        # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
        if cfe_state.timestep_rainfall_input_m > cfe_state.potential_et_m_per_timestep:
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.potential_et_m_per_timestep
            cfe_state.timestep_rainfall_input_m = cfe_state.timestep_rainfall_input_m.sub(cfe_state.actual_et_from_rain_m_per_timestep)

        # If rainfall is less than PET, all rainfall gets consumed as AET
        else: 
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.timestep_rainfall_input_m
            cfe_state.timestep_rainfall_input_m = torch.tensor(0.0, dtype=torch.float)
    
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_timestep - cfe_state.actual_et_from_rain_m_per_timestep
        
        return
                
                
    # __________________________________________________________________________________________________________
    ########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
    ##########                -or-                 ###############
    ##########    TWO OUTLET NONLINEAR RESERVOIR   ###############                        
    def gw_conceptual_reservoir_flux_calc(self, cfe_state, gw_reservoir):
        """
            This function calculates the flux from a linear, or nonlinear 
            conceptual reservoir with one or two outlets, or from an
            exponential nonlinear conceptual reservoir with only one outlet.
            In the non-exponential instance, each outlet can have its own
            activation storage threshold.  Flow from the second outlet is 
            turned off by setting the discharge coeff. to 0.0.
        """

        # This is basically only running for GW, so changed the variable name from primary_flux to primary_flux_from_gw_m to avoid confusion
        # if reservoir['is_exponential'] == True: 
        flux_exponential = torch.exp(gw_reservoir['exponent_primary'] *  gw_reservoir['storage_m'] / gw_reservoir['storage_max_m']) - torch.tensor(1.0, dtype=torch.float)
        cfe_state.primary_flux_from_gw_m = gw_reservoir['coeff_primary'] * flux_exponential
        cfe_state.secondary_flux_from_gw_m = torch.tensor(0.0, dtype=torch.float)
        return
    
    def soil_conceptual_reservoir_flux_calc(self, cfe_state, soil_reservoir):
        
        # Calculate the primary flux 
        storage_above_threshold_m = soil_reservoir['storage_m'] - soil_reservoir['storage_threshold_primary_m']
        
        if storage_above_threshold_m > 0.0:

            storage_diff = soil_reservoir['storage_max_m'] - soil_reservoir['storage_threshold_primary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = torch.pow(storage_ratio, soil_reservoir['exponent_primary'])
            
            cfe_state.primary_flux_m = soil_reservoir['coeff_primary'] * storage_power

            if cfe_state.primary_flux_m > storage_above_threshold_m:
                cfe_state.primary_flux_m = storage_above_threshold_m
        else:
            cfe_state.primary_flux_m = torch.tensor(0.0, dtype=torch.float)
                
        # Calculate the secondary flux 
        storage_above_threshold_m = soil_reservoir['storage_m'] - soil_reservoir['storage_threshold_secondary_m']
        
        if storage_above_threshold_m > 0.0:
            
            storage_diff = soil_reservoir['storage_max_m'] - soil_reservoir['storage_threshold_secondary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = torch.pow(storage_ratio, soil_reservoir['exponent_secondary'])
            
            cfe_state.secondary_flux_m = soil_reservoir['coeff_secondary'] * storage_power
            
            if cfe_state.secondary_flux_m > (storage_above_threshold_m - cfe_state.primary_flux_m):
                cfe_state.secondary_flux_m = storage_above_threshold_m - cfe_state.primary_flux_m
                
        else: 
            cfe_state.secondary_flux_m = torch.tensor(0.0, dtype=torch.float)
                
        return
    
    
    # __________________________________________________________________________________________________________
    #  SCHAAKE RUNOFF PARTITIONING SCHEME
    def Schaake_partitioning_scheme(self,cfe_state):
        """
            This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
            infiltration_depth_m using the scheme from Schaake et al. 1996. 
            !--------------------------------------------------------------------------------
            modified by FLO April 2020 to eliminate reference to ice processes, 
            and to de-obfuscate and use descriptive and dimensionally consistent variable names.
            
            inputs:
              timestep_d
              Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
              column_total_soil_moisture_deficit_m (soil_reservoir_storage_deficit_m)
              water_input_depth_m (timestep_rainfall_input_m) amount of water input to soil surface this time step [m]
            outputs:
              surface_runoff_depth_m      amount of water partitioned to surface water this time step [m]
              infiltration_depth_m
        """
        
        if 0 < cfe_state.timestep_rainfall_input_m:
            
            if 0 > cfe_state.soil_reservoir_storage_deficit_m:
                
                cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = torch.tensor(0.0, dtype=torch.float)
                
            else:
                
                schaake_exp_term = torch.exp( - cfe_state.Schaake_adjusted_magic_constant_by_soil_type * cfe_state.timestep_d)
                
                Schaake_parenthetical_term = (1.0 - schaake_exp_term)
                
                Ic = cfe_state.soil_reservoir_storage_deficit_m * Schaake_parenthetical_term
                
                Px = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = (Px * (Ic / (Px + Ic)))
                
                if 0.0 < (cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m):
                    
                    cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m
                    
                else:
                    
                    cfe_state.surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float)
                    
                cfe_state.infiltration_depth_m =  cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
                    
        else:
            
            cfe_state.surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float)
            cfe_state.infiltration_depth_m = torch.tensor(0.0, dtype=torch.float)
            
        return
    
    # __________________________________________________________________________________________________________
    def Xinanjiang_partitioning_scheme(self,cfe_state): 
        """
            This module takes the water_input_depth_m and separates it into surface_runoff_depth_m
            and infiltration_depth_m by calculating the saturated area and runoff based on a scheme developed
            for the Xinanjiang model by Jaywardena and Zhou (2000). According to Knoben et al.
            (2019) "the model uses a variable contributing area to simulate runoff.  [It] uses
            a double parabolic curve to simulate tension water capacities within the catchment, 
            instead of the original single parabolic curve" which is also used as the standard 
            VIC fomulation.  This runoff scheme was selected for implementation into NWM v3.0.
            REFERENCES:
            1. Jaywardena, A.W. and M.C. Zhou, 2000. A modified spatial soil moisture storage 
                capacity distribution curve for the Xinanjiang model. Journal of Hydrology 227: 93-113
            2. Knoben, W.J.M. et al., 2019. Supplement of Modular Assessment of Rainfall-Runoff Models
                Toolbox (MARRMoT) v1.2: an open-source, extendable framework providing implementations
                of 46 conceptual hydrologic models as continuous state-space formulations. Supplement of 
                Geosci. Model Dev. 12: 2463-2480.
            -------------------------------------------------------------------------
            Written by RLM May 2021
            Adapted by JMFrame September 2021 for new version of CFE
            Further adapted by QiyueL August 2022 for python version of CFE
            ------------------------------------------------------------------------
            Inputs
            double  time_step_rainfall_input_m           amount of water input to soil surface this time step [m]
            double  field_capacity_m                     amount of water stored in soil reservoir when at field capacity [m]
            double  max_soil_moisture_storage_m          total storage of the soil moisture reservoir (porosity*soil thickness) [m]
            double  column_total_soil_water_m     current storage of the soil moisture reservoir [m]
            double  a_inflection_point_parameter  a parameter
            double  b_shape_parameter             b parameter
            double  x_shape_parameter             x parameter
                //
            Outputs
            double  surface_runoff_depth_m        amount of water partitioned to surface water this time step [m]
            double  infiltration_depth_m          amount of water partitioned as infiltration (soil water input) this time step [m]
            ------------------------------------------------------------------------- 
        """

        # partition the total soil water in the column between free water and tension water
        free_water_m = cfe_state.soil_reservoir['storage_m']- cfe_state.soil_reservoir['storage_threshold_primary_m'];

        if (0.0 < free_water_m):

            tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m'];

        else: 

            free_water_m = torch.tensor(0.0, dtype=torch.float);
            tension_water_m = cfe_state.soil_reservoir['storage_m']
        
        # estimate the maximum free water and tension water available in the soil column
        max_free_water_m = cfe_state.soil_reservoir['storage_max_m'] - cfe_state.soil_reservoir['storage_threshold_primary_m']
        max_tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m']

        # check that the free_water_m and tension_water_m do not exceed the maximum and if so, change to the max value
        if(max_free_water_m < free_water_m): 
            free_water_m = max_free_water_m

        if(max_tension_water_m < tension_water_m): 
            tension_water_m = max_tension_water_m

        """
            NOTE: the impervious surface runoff assumptions due to frozen soil used in NWM 3.0 have not been included.
            We are assuming an impervious area due to frozen soils equal to 0 (see eq. 309 from Knoben et al).

            The total (pervious) runoff is first estimated before partitioning into surface and subsurface components.
            See Knoben et al eq 310 for total runoff and eqs 313-315 for partitioning between surface and subsurface
            components.

            Calculate total estimated pervious runoff. 
            NOTE: If the impervious surface runoff due to frozen soils is added,
            the pervious_runoff_m equation will need to be adjusted by the fraction of pervious area.
        """
        a_Xinanjiang_inflection_point_parameter = 1
        b_Xinanjiang_shape_parameter = 1
        x_Xinanjiang_shape_parameter = 1

        if ((tension_water_m/max_tension_water_m) <= (0.5 - a_Xinanjiang_inflection_point_parameter)): 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * \
                (torch.pow((0.5 - a_Xinanjiang_inflection_point_parameter),\
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        torch.pow((1.0 - (tension_water_m/max_tension_water_m)),\
                            b_Xinanjiang_shape_parameter))

        else: 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m* \
                (1.0 - torch.pow((0.5 + a_Xinanjiang_inflection_point_parameter), \
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        torch.pow((1.0 - (tension_water_m/max_tension_water_m)),\
                            (b_Xinanjiang_shape_parameter)))
    
        # Separate the surface water from the pervious runoff 
        ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
        ## the surface_runoff_depth_m.
        
        cfe_state.surface_runoff_depth_m = pervious_runoff_m * \
             (1.0 - torch.pow((1.0 - (free_water_m/max_free_water_m)),x_Xinanjiang_shape_parameter))

        # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
        # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
        if(cfe_state.surface_runoff_depth_m < 0.0): 
            cfe_state.surface_runoff_depth_m = torch.tensor(0.0, dtype=torch.float)
    
        # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
        # change the surface water runoff value to the water input depth.
        if(cfe_state.surface_runoff_depth_m > cfe_state.timestep_rainfall_input_m): 
             cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
        
        # Separate the infiltration from the total water input depth to the soil surface.
        cfe_state.infiltration_depth_m = cfe_state.timestep_rainfall_input_m- cfe_state.surface_runoff_depth_m;    

        return
                            
    # __________________________________________________________________________________________________________
    def et_from_soil(self,cfe_state):
        """
            Take AET from soil moisture storage, 
            using Budyko type curve to limit PET if wilting<soilmoist<field_capacity
        """
        
        if cfe_state.reduced_potential_et_m_per_timestep > 0:
            
            if cfe_state.soil_reservoir['storage_m'] >= cfe_state.soil_reservoir['storage_threshold_primary_m']:
            
                cfe_state.actual_et_from_soil_m_per_timestep = torch.minimum(cfe_state.reduced_potential_et_m_per_timestep, cfe_state.soil_reservoir['storage_m'])

            elif ((cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']) and 
                (cfe_state.soil_reservoir['storage_m'] < cfe_state.soil_reservoir['storage_threshold_primary_m'])):
            
                Budyko_numerator = cfe_state.soil_reservoir['storage_m'] - cfe_state.soil_reservoir['wilting_point_m']
                Budyko_denominator = cfe_state.soil_reservoir['storage_threshold_primary_m'] - cfe_state.soil_reservoir['wilting_point_m']
                Budyko = Budyko_numerator / Budyko_denominator

                cfe_state.actual_et_from_soil_m_per_timestep = torch.minimum(Budyko * cfe_state.reduced_potential_et_m_per_timestep,cfe_state.soil_reservoir['storage_m'])
            cfe_state.soil_reservoir['storage_m'] = cfe_state.soil_reservoir['storage_m'].sub(cfe_state.actual_et_from_soil_m_per_timestep)
            cfe_state.reduced_potential_et_m_per_timestep = cfe_state.reduced_potential_et_m_per_timestep.sub(cfe_state.actual_et_from_soil_m_per_timestep)
        
        return
            
            
    # __________________________________________________________________________________________________________
    # def check_is_fabs_less_than_epsilon(self,cfe_state,epsilon=1.0e-9):
    #     """ in the instance of calling the gw reservoir the secondary flux should be zero- verify
    #         From Line 157 of https://github.com/NOAA-OWP/cfe/blob/master/original_author_code/cfe.c
    #     """
    #     a = cfe_state.secondary_flux
    #     if np.abs(a) < epsilon: ##change to torch later
    #         cfe_state.is_fabs_less_than_epsilon = True
    #     else:
    #         print("problem with nonzero flux point 1\n")
    #         cfe_state.is_fabs_less_than_epsilon = False 
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    def soil_moisture_flux_calc_with_ode(self, cfe_state, reservoir):
        """
            This function solves the soil moisture mass balance.
            Inputs:
                reservoir
            Outputs:
                primary_flux_m (percolation)
                secondary_flux_m (lateral flow)
                actual_et_from_soil_m_per_timestep (et_from_soil)
        """
        
        # Initialization
        y0 = reservoir['storage_m']
        t = torch.tensor([0, 0.05, 0.15, 0.3, 0.6, 1.0]) # ODE time descritization of one time step

        # Pass parameters beforehand
        # device = 'cpu'
        func = soil_moisture_flux_ode(cfe_state=cfe_state, reservoir=reservoir).to(cfe_state.cfg.device)

        # Solve and ODE
        sol = odeint(
            func,
            y0,
            t,
            # atol=1e-5,
            # rtol=1e-5,
            # adjoint_params=()
        )

        # Finalize results
        ts_concat = t
        ys_concat = sol.squeeze()
        t_proportion = torch.diff(ts_concat, dim=0) # ts_concat[1:] - ts_concat[:-1]
        
        
        # Create the kernel tensor with torch.ones
        kernel = torch.ones(2)

        # Get the moving average y values in between the time intervals
        convolved = F.conv1d(ys_concat.unsqueeze(0).unsqueeze(0), kernel.float().unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        # Divide by 2 to match np.convolve
        ys_avg_ = convolved / 2
        ys_avg = ys_avg_[1:-1]
        
        # Get each flux values and scale it
        lateral_flux = torch.zeros(ys_avg.shape)
        perc_lat_switch = ys_avg - reservoir['storage_threshold_primary_m'] > 0
        lateral_flux[perc_lat_switch] = reservoir['coeff_secondary'] * torch.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), torch.tensor([1.0]))
        lateral_flux_frac = lateral_flux * t_proportion

        perc_flux = torch.zeros(ys_avg.shape)
        perc_flux[perc_lat_switch] = reservoir['coeff_primary'] * torch.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), torch.tensor([1.0]))
        perc_flux_frac = perc_flux * t_proportion

        et_from_soil = torch.zeros(ys_avg.shape)
        ET_switch = ys_avg - reservoir['wilting_point_m'] > 0
        et_from_soil[ET_switch] = cfe_state.reduced_potential_et_m_per_timestep * torch.minimum(
            (ys_avg[ET_switch] - reservoir['wilting_point_m']) / (reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']), torch.tensor([1.0]))
        et_from_soil_frac = et_from_soil * t_proportion
        
        infilt_to_soil = torch.tensor(cfe_state.infiltration_depth_m).repeat(ys_avg.shape)
        infilt_to_soil_frac = infilt_to_soil * t_proportion

        # Scale fluxes (Since the sum of all the estimated flux above usually exceed the input flux because of calculation errors, scale it
        # The more finer ODE time descritization you use, the less errors you get, but the more calculation time it takes 
        sum_outflux = lateral_flux_frac + perc_flux_frac + et_from_soil_frac
        if sum_outflux.any() == 0:
            flux_scale = 0
        else:
            flux_scale = torch.zeros(infilt_to_soil_frac.shape)
            flux_scale[sum_outflux != 0] = (torch.diff(-ys_concat, dim=0)[sum_outflux != 0] + infilt_to_soil_frac[
                sum_outflux != 0]) / sum_outflux[sum_outflux != 0]
            flux_scale[sum_outflux == 0] = 0
        scaled_lateral_flux = lateral_flux_frac * flux_scale
        scaled_perc_flux = perc_flux_frac * flux_scale
        scaled_et_flux = et_from_soil_frac * flux_scale

        # Pass the results
        # ? Do these all gets tracked? 
        cfe_state.primary_flux_m = math.fsum(scaled_perc_flux)
        cfe_state.secondary_flux_m = math.fsum(scaled_lateral_flux)
        cfe_state.actual_et_from_soil_m_per_timestep = math.fsum(scaled_et_flux)
        reservoir['storage_m'] = ys_concat[-1]

        # print(f'primary_flux_m: {primary_flux_m}')
        # print(f'secondary_flux_m: {secondary_flux_m}')
        # print(f'actual_et_from_soil_m_per_timestep: {actual_et_from_soil_m_per_timestep}')
        # print(f'reservoir["storage_m"]: {reservoir["storage_m"]}')

        return