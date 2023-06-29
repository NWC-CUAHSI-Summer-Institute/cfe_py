import time
import numpy as np
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
import cfe
import copy

class BMI_CFE():
    def __init__(self, cfg_file=None):
        
        # ________________________________________________
        # Create a Bmi CFE model that is ready for initialization
        
        super(BMI_CFE, self).__init__()
        self._values = {}
        self._values_unperturbed = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        
        # ________________________________________________
        # Required, static attributes of the model

        self._att_map = {
            'model_name':         'Conceptual Functional Equivalent (CFE)',
            'version':            '1.0',
            'author_name':        'Jonathan Martin Frame',
            'grid_type':          'scalar',
            'time_step_size':      3600, 
            'time_units':         '1 hour' }
    
        # ________________________________________________
        # Input variable names (CSDMS standard names)

        self._input_var_names = [
            'atmosphere_water__time_integral_of_precipitation_mass_flux',
            'water_potential_evaporation_flux']
    
        # ________________________________________________
        # Output variable names (CSDMS standard names)

        self._output_var_names = ['land_surface_water__runoff_depth', 
                                  'land_surface_water__runoff_volume_flux',
                                  "DIRECT_RUNOFF",
                                  "GIUH_RUNOFF",
                                  "NASH_LATERAL_RUNOFF",
                                  "DEEP_GW_TO_CHANNEL_FLUX",
                                  "SOIL_CONCEPTUAL_STORAGE",
                                  "atmosphere_water__time_integral_of_precipitation_mass_flux"]
        
        # ________________________________________________
        # Create a Python dictionary that maps CSDMS Standard
        # Names to the model's internal variable names.
        # This is going to get long, 
        #     since the input variable names could come from any forcing...

        self._var_name_units_map = {
                                'land_surface_water__runoff_volume_flux':['flux_Qout_m','m3 h-1'],
                                'land_surface_water__runoff_depth':['total_discharge','m h-1'],
                                #--------------   Dynamic inputs --------------------------------
                                'atmosphere_water__time_integral_of_precipitation_mass_flux':['timestep_rainfall_input_m','m h-1'],
                                'water_potential_evaporation_flux':['potential_et_m_per_s','m s-1'],
                                'DIRECT_RUNOFF':['surface_runoff_depth_m','m'],
                                'GIUH_RUNOFF':['flux_giuh_runoff_m','m'],
                                'NASH_LATERAL_RUNOFF':['flux_nash_lateral_runoff_m','m'],
                                'DEEP_GW_TO_CHANNEL_FLUX':['flux_from_deep_gw_to_chan_m','m'],
                                'SOIL_CONCEPTUAL_STORAGE':["soil_reservoir['storage_m']", 'm']
                                }

        # ________________________________________________
        # this is the bmi configuration file
        self.cfg_file = cfg_file

    #__________________________________________________________________
    #__________________________________________________________________
    # BMI: Model Control Function
    def initialize(self,current_time_step=0):
        self.current_time_step=current_time_step

        # ________________________________________________
        # Create some lookup tabels from the long variable names
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for long_name in self._var_name_units_map.keys()}
        
        # ________________________________________________
        # Initalize all the variables
        # so that they'll be picked up with the get functions
        for long_var_name in list(self._var_name_units_map.keys()):
            
            # All the variables are single values
            # so just set to zero for now
            self._values[long_var_name] = 0
            setattr( self, self.get_var_name(long_var_name), 0 )

        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()                                    #
        
        # ________________________________________________
        # Derive some info about the ensembles from the configurations
        self.n_cfe_ensembles = np.int32(np.max([1, 
                                self.perturb_forcings["N"] * self.perturb_states["N"]]))
        self.ensemble_member_list = list(range(self.n_cfe_ensembles))
        
        
        # ________________________________________________
        # The configuration should let the BMI know what mode to run in (framework vs standalone)
        # If it is stand alone, then load in the forcing and read the time from the forcig file
        if self.stand_alone == 1:
            self.load_forcing_file()
            self.current_time = pd.to_datetime(self.forcing_data['time'][self.current_time_step])

        # ________________________________________________
        # In order to check mass conservation at any time
        self.reset_volume_tracking()
        
        # ________________________________________________
        # initialize simulation constants
        atm_press_Pa=101325.0
        unit_weight_water_N_per_m3=9810.0
        
        # ________________________________________________
        # Time control
        self.time_step_size = 3600
        self.timestep_h = self.time_step_size / 3600
        self.timestep_d = self.timestep_h / 24.0
        self.current_time_step = 0
        self.current_time = pd.Timestamp(year=2007, month=10, day=1, hour=0)
        
        # ________________________________________________
        # Inputs
        self.E_timestep_rainfall_input_m = [0 for ens in self.ensemble_member_list]
        self.E_potential_et_m_per_s      = [0 for ens in self.ensemble_member_list]
        
        # ________________________________________________
        # calculated flux variables
        # surface runoff that goes through the GIUH convolution process
        self.E_flux_overland_m                = [0 for ens in self.ensemble_member_list]
        # flux from soil to deeper groundwater reservoir
        self.E_flux_perc_m                    = [0 for ens in self.ensemble_member_list]
        # lateral flux in the subsurface to the Nash cascade
        self.E_flux_lat_m                     = [0 for ens in self.ensemble_member_list]
        # flux from the deep reservoir into the channels
        self.E_flux_from_deep_gw_to_chan_m    = [0 for ens in self.ensemble_member_list]
        # the available space in the conceptual groundwater reservoir
        self.E_gw_reservoir_storage_deficit_m = [0 for ens in self.ensemble_member_list]
        self.E_primary_flux                   = [0 for ens in self.ensemble_member_list] # temporary vars.
        self.E_secondary_flux                 = [0 for ens in self.ensemble_member_list] # temporary vars.
        self.E_total_discharge                = [0 for ens in self.ensemble_member_list]
        # Added by Ryoko for soil-ode
        self.E_diff_infilt                    = [0 for ens in self.ensemble_member_list]
        self.E_diff_perc                      = [0 for ens in self.ensemble_member_list] 
        # ________________________________________________
        # Evapotranspiration
        self.E_potential_et_m_per_timestep = [0 for ens in self.ensemble_member_list]
        self.E_actual_et_m_per_timestep    = [0 for ens in self.ensemble_member_list]
        # Added by Ryoko for soil-ode
        self.E_reduced_potential_et_m_per_timestep = [0 for ens in self.ensemble_member_list]
        self.E_actual_et_from_rain_m_per_timestep = [0 for ens in self.ensemble_member_list]
        self.E_actual_et_from_soil_m_per_timestep = [0 for ens in self.ensemble_member_list]
        self.E_nash_storage = [copy.deepcopy(self.nash_storage) for ens in self.ensemble_member_list]
        # ________________________________________________________
        # Set these values now that we have the information from the configuration file.
        self.E_runoff_queue_m_per_timestep = [np.zeros(len(self.giuh_ordinates)+1) for ens in self.ensemble_member_list]
        self.E_num_giuh_ordinates = [len(self.giuh_ordinates) for ens in self.ensemble_member_list]
        self.E_num_lateral_flow_nash_reservoirs = [self.E_nash_storage[ens].shape[0] for ens in self.ensemble_member_list]
        
        # ________________________________________________
        # Local values to be used in setting up soil reservoir
        trigger_z_m = 0.5
        field_capacity_atm_press_fraction = self.alpha_fc
        
        # ________________________________________________
        # ________________________________________________
        # SOIL RESERVOIR CONFIGURATION
        
        # ________________________________________________
        # Soil outflux calculation, Equation 3 in Fred Ogden's document
        
        H_water_table_m = field_capacity_atm_press_fraction * atm_press_Pa / unit_weight_water_N_per_m3 
        
        soil_water_content_at_field_capacity = self.soil_params['smcmax'] * \
                        np.power(H_water_table_m/self.soil_params['satpsi'],(1.0/self.soil_params['bb'])) 
        
        Omega = H_water_table_m - trigger_z_m
        
        # ________________________________________________
        # Upper & lower limit of the integral in Equation 4 in Fred Ogden's document
        
        lower_lim = np.power(Omega, (1.0-1.0/self.soil_params['bb']))/(1.0-1.0/self.soil_params['bb'])
        
        upper_lim = np.power(Omega+self.soil_params['D'],(1.0-1.0/self.soil_params['bb']))/(1.0-1.0/self.soil_params['bb'])

        # ________________________________________________
        # Integral & power term in Equation 4 & 5 in Fred Ogden's document
        
        storage_thresh_pow_term = np.power(1.0/self.soil_params['satpsi'],(-1.0/self.soil_params['bb']))

        lim_diff = (upper_lim-lower_lim)

        field_capacity_storage_threshold_m = self.soil_params['smcmax'] * storage_thresh_pow_term * lim_diff
        
        # ________________________________________________
        # lateral flow function parameters
        assumed_near_channel_water_table_slope = 0.01 # [L/L]
        lateral_flow_threshold_storage_m       = field_capacity_storage_threshold_m # Equation 4 & 5  in Fred Ogden's document
#         lateral_flow_linear_reservoir_constant = 2.0 * assumed_near_channel_water_table_slope * \     # Not used
#                                                  self.soil_params['mult'] * NWM_soil_params.satdk * \ # Not used
#                                                  self.soil_params['D'] * drainage_density_km_per_km2  # Not used
#         lateral_flow_linear_reservoir_constant *= 3600.0                                              # Not used
        self.E_soil_reservoir_storage_deficit_m  = [0 for ens in self.ensemble_member_list]

        # ________________________________________________
        # Subsurface reservoirs
        
        self.E_volstart        = [0 for ens in self.ensemble_member_list]
        self.E_vol_in_gw_start = [0 for ens in self.ensemble_member_list]
        self.E_vol_soil_start  = [0 for ens in self.ensemble_member_list]
        # Set dictionaries of the groundwater and soil reservoir ensembles. Will be filled in below.
        self.E_gw_reservoir = {}
        self.E_soil_reservoir = {}
        # - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        for ens in self.ensemble_member_list:
            self.E_gw_reservoir[ens] = {'is_exponential':True,
                                          'storage_max_m':self.max_gw_storage,
                                          'coeff_primary':self.Cgw,
                                          'exponent_primary':self.expon,
                                          'storage_threshold_primary_m':0.0,
                              # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off. 
                                          'storage_threshold_secondary_m':0.0,
                                          'coeff_secondary':0.0,
                                          'exponent_secondary':1.0}
        
            self.E_gw_reservoir[ens]['storage_m'] = self.E_gw_reservoir[ens]['storage_max_m'] * 0.01
            self.E_volstart[ens] += self.E_gw_reservoir[ens]['storage_m']

            self.E_soil_reservoir[ens] = {'is_exponential':False,
                                            'wilting_point_m':self.soil_params['wltsmc'] * self.soil_params['D'],
                                            'storage_max_m':self.soil_params['smcmax'] * self.soil_params['D'],
                                            # Controls percolation to GW, Equation 11
                                            'coeff_primary':self.soil_params['satdk'] * self.soil_params['slop'] * self.time_step_size,
                                            # Controls percolation to GW, FIXED to 1 based on Equation 11
                                            'exponent_primary':1.0,                                                                     
                                            'storage_threshold_primary_m': field_capacity_storage_threshold_m,                         
                                            # Controls lateral flow
                                            'coeff_secondary':self.K_lf,                                                                
                                            # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
                                            'exponent_secondary':1.0,                                                                  
                                            'storage_threshold_secondary_m':lateral_flow_threshold_storage_m}
            self.E_soil_reservoir[ens]['storage_m'] = self.E_soil_reservoir[ens]['storage_max_m'] * 0.667
            
            self.E_volstart[ens] += self.E_soil_reservoir[ens]['storage_m']
        
        # ________________________________________________
        # Schaake partitioning 
        self.refkdt = 3.0
        self.Schaake_adjusted_magic_constant_by_soil_type = self.refkdt * self.soil_params['satdk'] / 2.0e-06
        self.Schaake_output_runoff_m = 0
        self.infiltration_depth_m = 0
        
        # ________________________________________________
        # Nash cascade        
        self.K_nash = 0.03 #Default value, but should be set in configuration file

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s    
        self.output_factor_cms =  (1/1000) * (self.catchment_area_km2 * 1000*1000) * (1/3600)

        # ________________________________________________
        # Initalize all the variables as dictionaries, so we can fill in each ensemble
        for var_name in self._var_name_units_map.keys():
            self._values[var_name] = {}
            

        #############################################################################
        # _________________________________________________________________________ #
        # _________________________________________________________________________ #
        # CREATE AN INSTANCE OF THE CONCEPTUAL FUNCTIONAL EQUIVALENT MODEL, or many #
        self.cfe_models = {i:cfe.CFE() for i in self.ensemble_member_list}
        # _________________________________________________________________________ #
        #############################################################################
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update(self):
        
        # Since whatever has been set before update should be remembered
        # specifically the forcings, because we want to perturb based on the origional value
        perturb_this_forcing = 'atmosphere_water__time_integral_of_precipitation_mass_flux'
        self.make_a_copy_of_unperturbed_value(perturb_this_forcing)
        
        self.current_ensemble_member = 0
        for forcing_ens in range(self.perturb_forcings["N"]):
            
            # This perturbs the precipitation from the saved unperturbed precip
            self.perturb_forcing_from_unperturbed_value(perturb_this_forcing)
            
            for state_ens in range(self.perturb_states["N"]):
            
                ens = self.current_ensemble_member
                
                self.set_ensemble_member_precipitation()
                
                self.E_volin[ens] += self.E_timestep_rainfall_input_m[ens]
                
                self.perturb_cfe_states()

                self.set_current_cfe_state_values_from_ensemble()

                self.cfe_models[ens].run_cfe(self)
                
                self.set_output()

                self.fill_ensemble_array_from_current_cfe_state()
                
                self.current_ensemble_member += 1
        
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update_until(self, until):
        
        for i in range(self.current_time_step, until):
                
            self.update()
        
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def finalize(self):

        self.finalize_mass_balance()
        self.reset_volume_tracking()

        """Finalize model."""
        self.cfe_model = None
        self.cfe_state = None
    
    # ________________________________________________
    # Mass balance tracking
    def reset_volume_tracking(self):
        self.E_volstart             = [0 for ens in self.ensemble_member_list]
        self.E_vol_et_from_soil     = [0 for ens in self.ensemble_member_list]
        self.E_vol_et_from_rain     = [0 for ens in self.ensemble_member_list]
        self.E_vol_partition_runoff = [0 for ens in self.ensemble_member_list]
        self.E_vol_partition_infilt = [0 for ens in self.ensemble_member_list]
        self.E_vol_out_giuh         = [0 for ens in self.ensemble_member_list]
        self.E_vol_end_giuh         = [0 for ens in self.ensemble_member_list]
        self.E_vol_to_gw            = [0 for ens in self.ensemble_member_list]
        self.E_vol_to_gw_start      = [0 for ens in self.ensemble_member_list]
        self.E_vol_to_gw_end        = [0 for ens in self.ensemble_member_list]
        self.E_vol_from_gw          = [0 for ens in self.ensemble_member_list]
        self.E_vol_in_nash          = [0 for ens in self.ensemble_member_list]
        self.E_vol_in_nash_end      = [0 for ens in self.ensemble_member_list]
        self.E_vol_out_nash         = [0 for ens in self.ensemble_member_list]
        self.E_vol_soil_start       = [0 for ens in self.ensemble_member_list]
        self.E_vol_to_soil          = [0 for ens in self.ensemble_member_list]
        self.E_vol_soil_to_lat_flow = [0 for ens in self.ensemble_member_list]
        self.E_vol_soil_to_gw       = [0 for ens in self.ensemble_member_list]
        self.E_vol_soil_end         = [0 for ens in self.ensemble_member_list]
        self.E_volin                = [0 for ens in self.ensemble_member_list]
        self.E_volout               = [0 for ens in self.ensemble_member_list]
        self.E_volend               = [0 for ens in self.ensemble_member_list]
        # Added by Ryoko for soil-ode
        self.E_vol_partition_runoff_IOF   = [0 for ens in self.ensemble_member_list]
        self.E_vol_partition_runoff_SOF   = [0 for ens in self.ensemble_member_list]
        self.E_vol_et_to_atm        = [0 for ens in self.ensemble_member_list]
        self.E_vol_et_from_soil     = [0 for ens in self.ensemble_member_list]
        self.E_vol_PET              = [0 for ens in self.ensemble_member_list]
        return
    
    #________________________________________________________
    def config_from_json(self):
        with open(self.cfg_file) as data_file:
            data_loaded = json.load(data_file)

        # ___________________________________________________
        ## MANDATORY CONFIGURATIONS
        self.forcing_file               = data_loaded['forcing_file']
        self.catchment_area_km2         = data_loaded['catchment_area_km2']
        
        # Soil parameters
        self.alpha_fc                   = data_loaded['alpha_fc']
        self.soil_params                = {}
        self.soil_params['bb']          = data_loaded['soil_params']['bb']
        self.soil_params['D']           = data_loaded['soil_params']['D']
        self.soil_params['satdk']       = data_loaded['soil_params']['satdk']
        self.soil_params['satpsi']      = data_loaded['soil_params']['satpsi']
        self.soil_params['slop']        = data_loaded['soil_params']['slop']
        self.soil_params['smcmax']      = data_loaded['soil_params']['smcmax']
        self.soil_params['wltsmc']      = data_loaded['soil_params']['wltsmc']
        self.K_lf                       = data_loaded['K_lf']
        self.soil_params['scheme']      = data_loaded['soil_scheme']
        
        # Groundwater parameters
        self.max_gw_storage             = data_loaded['max_gw_storage']
        self.Cgw                        = data_loaded['Cgw']
        self.expon                      = data_loaded['expon']
        
        # Other modules 
        self.K_nash                     = data_loaded['K_nash']
        self.nash_storage               = np.array(data_loaded['nash_storage'])
        self.giuh_ordinates             = np.array(data_loaded['giuh_ordinates'])
        
        # Partitioning parameters
        self.surface_partitioning_scheme= data_loaded['partition_scheme']
        
        # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        # For data assimilation, or other non-deterministic type of modes
        d_temp = data_loaded["perturb_forcings_mean_std_N"]
        self.perturb_forcings = {"mean":d_temp[0], "std":d_temp[1], "N":d_temp[2]}
        d_temp = data_loaded["perturb_states_mean_std_N"]
        self.perturb_states = {"mean":d_temp[0], "std":d_temp[1], "N":d_temp[2]}

        
        # ___________________________________________________
        # OPTIONAL CONFIGURATIONS
        if 'stand_alone' in data_loaded.keys():
            self.stand_alone                    = data_loaded['stand_alone']
        if 'forcing_file' in data_loaded.keys():
            self.reads_own_forcing              = True
            self.forcing_file                   = data_loaded['forcing_file']
        if 'unit_test' in data_loaded.keys():
            self.unit_test                      = data_loaded['unit_test']
            self.compare_results_file           = data_loaded['compare_results_file']
        # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        # Soil representation selection
        if 'soil_scheme' in data_loaded.keys():
            self.soil_scheme = data_loaded["soil_scheme"]
        else:
            self.soil_scheme = 'classic' 
        
        self.verbose = data_loaded['verbose']
        if self.verbose == 'False':
            self.verbose = False
        elif self.verbose == 'True':
            self.verbose = True
        return

    
    #________________________________________________________        
    def finalize_mass_balance(self):
        
        for ens in range(self.n_cfe_ensembles):
        
            self.E_volend[ens]        = self.E_soil_reservoir[ens]['storage_m'] + self.E_gw_reservoir[ens]['storage_m']
            self.vol_in_gw_end = self.E_gw_reservoir[ens]['storage_m']

            # the GIUH queue might have water in it at the end of the simulation, so sum it up.
            self.E_vol_end_giuh[ens] = np.sum(self.E_runoff_queue_m_per_timestep[ens])
            self.E_vol_in_nash_end[ens] = np.sum(self.E_nash_storage[ens])

            self.E_vol_soil_end[ens] = self.E_soil_reservoir[ens]['storage_m']

            self.global_residual  = self.E_volstart[ens] + self.E_volin[ens] - self.E_volout[ens] - self.E_volend[ens] - self.E_vol_end_giuh[ens]
            self.partition_residual = self.E_volin[ens] - self.E_vol_partition_runoff[ens] - \
                                      self.E_vol_partition_infilt[ens] - self.E_vol_et_from_rain[ens]
            self.giuh_residual    = self.E_vol_partition_runoff[ens] - self.E_vol_out_giuh[ens] - self.E_vol_end_giuh[ens]
            self.soil_residual    = self.E_vol_soil_start[ens] + self.E_vol_to_soil[ens] - \
                                    self.E_vol_soil_to_lat_flow[ens]  - self.E_vol_to_gw[ens] - \
                                    self.E_vol_et_from_soil[ens] - self.E_vol_soil_end[ens]
            self.nash_residual    = self.E_vol_in_nash[ens] - self.E_vol_out_nash[ens] - self.E_vol_in_nash_end[ens]
            self.gw_residual      = self.E_vol_in_gw_start[ens] + self.E_vol_to_gw[ens] - self.E_vol_from_gw[ens] - self.vol_in_gw_end

            if self.verbose:            
                print("\nGLOBAL MASS BALANCE")
                print("  initial volume: {:8.4f}".format(self.E_volstart[ens]))
                print("    volume input: {:8.4f}".format(self.E_volin[ens]))
                print("   volume output: {:8.4f}".format(self.E_volout[ens]))
                print("    final volume: {:8.4f}".format(self.E_volend[ens]))
                print("        residual: {:6.4e}".format(self.global_residual))


                print("\nPARTITION MASS BALANCE")
                print("    surface runoff: {:8.4f}".format(self.E_vol_partition_runoff[ens]))
                print("      infiltration: {:8.4f}".format(self.E_vol_partition_infilt[ens]))
                print(" vol. et from rain: {:8.4f}".format(self.E_vol_et_from_rain[ens])) 
                print("partition residual: {:6.4e}".format(self.partition_residual))  

                print("\nGIUH MASS BALANCE");
                print("  vol. into giuh: {:8.4f}".format(self.E_vol_partition_runoff[ens]))
                print("   vol. out giuh: {:8.4f}".format(self.E_vol_out_giuh[ens]))
                print(" vol. end giuh q: {:8.4f}".format(self.E_vol_end_giuh[ens]))
                print("   giuh residual: {:6.4e}".format(self.giuh_residual))

                if self.soil_scheme == 'classic':
                    print("\nSOIL WATER CONCEPTUAL RESERVOIR MASS BALANCE")
                elif self.soil_scheme == 'ode':
                    print("\nSOIL WATER MASS BALANCE")
                print("     init soil vol: {:8.4f}".format(self.E_vol_soil_start[ens]))     
                print("    vol. into soil: {:8.4f}".format(self.E_vol_to_soil[ens]))
                print("  vol.soil2latflow: {:8.4f}".format(self.E_vol_soil_to_lat_flow[ens]))
                print("   vol. soil to gw: {:8.4f}".format(self.E_vol_soil_to_gw[ens]))
                print(" vol. et from soil: {:8.4f}".format(self.E_vol_et_from_soil[ens]))
                print("   final vol. soil: {:8.4f}".format(self.E_vol_soil_end[ens]))   
                print("  vol. soil resid.: {:6.4e}".format(self.soil_residual))

                print("\nNASH CASCADE CONCEPTUAL RESERVOIR MASS BALANCE")
                print("    vol. to nash: {:8.4f}".format(self.E_vol_in_nash[ens]))
                print("  vol. from nash: {:8.4f}".format(self.E_vol_out_nash[ens]))
                print(" final vol. nash: {:8.4f}".format(self.E_vol_in_nash_end[ens]))
                print("nash casc resid.: {:6.4e}".format(self.nash_residual))


                print("\nGROUNDWATER CONCEPTUAL RESERVOIR MASS BALANCE")
                print("init gw. storage: {:8.4f}".format(self.E_vol_in_gw_start[ens]))
                print("       vol to gw: {:8.4f}".format(self.E_vol_to_gw[ens]))
                print("     vol from gw: {:8.4f}".format(self.E_vol_from_gw[ens]))
                print("final gw.storage: {:8.4f}".format(self.vol_in_gw_end))
                print("    gw. residual: {:6.4e}".format(self.gw_residual))


            return
    
    #________________________________________________________ 
    def load_forcing_file(self):
        self.forcing_data = pd.read_csv(self.forcing_file)
        
    #________________________________________________________ 
    def load_unit_test_data(self):
        self.unit_test_data = pd.read_csv(self.compare_results_file)
        self.cfe_output_data = pd.DataFrame().reindex_like(self.unit_test_data)
        
    #________________________________________________________ 
    def run_unit_test(self, plot_lims=list(range(490, 550)), plot=False, print_fluxes=True):
        
        self.load_forcing_file()
        self.load_unit_test_data()
        
        self.current_time = pd.Timestamp(self.forcing_data['time'][0])
        
        for t, precipitation_input in enumerate(self.forcing_data['precip_rate']*3600):
            
            self.timestep_rainfall_input_m          = precipitation_input
            self.cfe_output_data.loc[t,'Time']      = self.current_time
            self.cfe_output_data.loc[t,'Time Step'] = self.current_time_step
            self.cfe_output_data.loc[t,'Rainfall']  = self.timestep_rainfall_input_m

            self.update()
            
            self.cfe_output_data.loc[t,'Direct Runoff']   = self.surface_runoff_depth_m
            self.cfe_output_data.loc[t,'GIUH Runoff']     = self.flux_giuh_runoff_m
            self.cfe_output_data.loc[t,'Lateral Flow']    = self.flux_nash_lateral_runoff_m
            self.cfe_output_data.loc[t,'Base Flow']       = self.flux_from_deep_gw_to_chan_m
            self.cfe_output_data.loc[t,'Total Discharge'] = self.total_discharge
            self.cfe_output_data.loc[t,'Flow']            = self.flux_Qout_m
            
            if self.soil_scheme.lower() == 'ode':
                self.cfe_output_data[t, 'SM storage']             = self.soil_reservoir['storage_m'] 
                self.cfe_output_data['Soil Moisture Content']       =  self.soil_reservoir['storage_m']/self.soil_params['D']
            
            if print_fluxes:
                print('{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},'.format(self.current_time, self.timestep_rainfall_input_m,
                                           self.surface_runoff_depth_m, self.flux_giuh_runoff_m, self.flux_nash_lateral_runoff_m,
                                           self.flux_from_deep_gw_to_chan_m, self.flux_Qout_m, self.total_discharge))
        
        if plot:
            
            outputs = ['Direct Runoff', 'GIUH Runoff', 'Lateral Flow', 'Base Flow', 'Total Discharge', 'Flow']
            if self.soil_scheme.lower() == 'ode':
                outputs.append('Soil Moisture Content')
            
            for output_type in outputs:
                fig,ax = plt.subplots(figsize = (8,6))
                
                l1, = ax.plot(self.cfe_output_data['Rainfall'][plot_lims], label='precipitation', c='gray', lw=.3)
                ax.set_ylabel('Precipitation')
                
                ax2 = ax.twinx()
                l2, = ax2.plot(self.cfe_output_data[output_type][plot_lims], label='cfe '+output_type)
                plot_handles = [l1, l2]
                if output_type in list(self.unit_test_data.keys()): 
                    l3, = ax2.plot(self.unit_test_data[output_type][plot_lims], '--', label='t-shirt '+output_type)
                    plot_handles.append(l3)
                # TODO: Check why T-shirt Flow appears to be the same values as T-shirt total discharge
                ax2.set_ylabel('Simulations')
                
                plt.legend(handles = [l1,l2,l3])
                plt.show()
                plt.close()

    #------------------------------------------------------------ 
    def set_output(self):
            
        ens = self.current_ensemble_member
        
        self._values['land_surface_water__runoff_depth'][ens] = self.flux_Qout_m
        self._values['land_surface_water__runoff_volume_flux'][ens] = self.total_discharge 
        self._values["DIRECT_RUNOFF"][ens] = self.surface_runoff_depth_m
        self._values["GIUH_RUNOFF"][ens] = self.flux_giuh_runoff_m
        self._values["NASH_LATERAL_RUNOFF"][ens] = self.flux_nash_lateral_runoff_m
        self._values["DEEP_GW_TO_CHANNEL_FLUX"][ens] = self.flux_from_deep_gw_to_chan_m
        self._values["SOIL_CONCEPTUAL_STORAGE"][ens] = self.soil_reservoir['storage_m']
        self._values["atmosphere_water__time_integral_of_precipitation_mass_flux"] = self.timestep_rainfall_input_m
            
    #---------------------------------------------------------------------------- 
    def initialize_forcings(self):
        for forcing_name in self.cfg_train['dynamic_inputs']:
            setattr(self, self._var_name_map_short_first[forcing_name], 0)

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names
    
    #------------------------------------------------------------ 
    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute( 'model_name' ) #JG Edit

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
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

    #-------------------------------------------------------------------
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

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
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
        return self.get_value_ptr(long_var_name)  #.dtype
    
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # JG Edit
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id  

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
#        return np.dtype(self.get_var_type(name)).itemsize
        return np.array(self.get_value(name)).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        # JG Edit
        # all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    #-------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return self._start_time #JG Edit

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return self._end_time #JG Edit


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.current_time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.get_attribute( 'time_step_size' ) #JG: Edit

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        setattr( self, self.get_var_name(var_name), value )
        self._values[var_name] = value
        
    #-------------------------------------------------------------------
    def make_a_copy_of_unperturbed_value(self, var_name):
        self._values_unperturbed[var_name] = copy.deepcopy(self._values[var_name])
        
    #-------------------------------------------------------------------
    def perturb_forcing_from_unperturbed_value(self, var_name):
        perturbation = np.random.normal(self.perturb_forcings["mean"], self.perturb_forcings["std"])
        unperturbed_value = copy.deepcopy(self._values_unperturbed[var_name])
        perturbed_forcing = unperturbed_value * perturbation
        self._values[var_name] = perturbed_forcing
        
    def set_ensemble_member_precipitation(self):
        var_name = "atmosphere_water__time_integral_of_precipitation_mass_flux"
        self.E_timestep_rainfall_input_m[self.current_ensemble_member] = self._values[var_name]
        
    #-------------------------------------------------------------------
    def perturb_cfe_states(self):
        ens = self.current_ensemble_member
        perturbation = np.random.normal(self.perturb_states["mean"], self.perturb_states["std"])
        self.E_soil_reservoir[ens]['storage_m'] = self.E_soil_reservoir[ens]['storage_m'] * perturbation
        perturbation = np.random.normal(self.perturb_states["mean"], self.perturb_states["std"])
        self.E_gw_reservoir[ens]['storage_m'] = self.E_gw_reservoir[ens]['storage_m'] * perturbation                                          
        
    #------------------------------------------------------------ 
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

        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(name)).flatten().shape[0] == 1:
            self.set_value(name, src)
        else:
            # JMFrame: Need to set the value with the updated array with new index value
            val = self.get_value_ptr(name)
            for i in inds.shape:
                val.flatten()[inds[i]] = src[i]
            self.set_value(name, val)

    #------------------------------------------------------------ 
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

    #------------------------------------------------------------ 
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
        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(var_name)).flatten().shape[0] == 1:
            return self.get_value(var_name)
        else:
            val_array = self.get_value(var_name).flatten()
            return np.array([val_array[i] for i in indices])

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------ 
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
 
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0: 
            return 1

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
       
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id=0):

        # JG Edit
        # 0 is the only id we have        
        if grid_id == 0:
            return 'scalar'

    #------------------------------------------------------------ 
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    #------------------------------------------------------------ 
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    #------------------------------------------------------------ 
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z")

    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    def set_current_cfe_state_values_from_ensemble(self):
        
        #----------------------------------#
        ens = self.current_ensemble_member #
        #----------------------------------#
        
        self.soil_reservoir = self.E_soil_reservoir[ens]
        self.gw_reservoir = self.E_gw_reservoir[ens]
        
        # ________________________________________________
        # Inputs
        self.timestep_rainfall_input_m = self.E_timestep_rainfall_input_m[ens]
        self.potential_et_m_per_s = self.E_potential_et_m_per_s[ens]
        
        # ________________________________________________
        # calculated flux variables
        # surface runoff that goes through the GIUH convolution process
        self.flux_overland_m = self.E_flux_overland_m[ens]
        # flux from soil to deeper groundwater reservoir
        self.flux_perc_m = self.E_flux_perc_m[ens]
        # lateral flux in the subsurface to the Nash cascade
        self.flux_lat_m = self.E_flux_lat_m[ens]
        # flux from the deep reservoir into the channels
        self.flux_from_deep_gw_to_chan_m = self.E_flux_from_deep_gw_to_chan_m[ens]
        # the available space in the conceptual groundwater reservoir
        self.gw_reservoir_storage_deficit_m = self.E_gw_reservoir_storage_deficit_m[ens]
        self.primary_flux = self.E_primary_flux[ens]
        self.secondary_flux = self.E_secondary_flux[ens]
        self.total_discharge = self.E_total_discharge[ens]
        # Added by Ryoko for soil-ode
        self.diff_infilt = self.E_diff_infilt[ens]
        self.diff_perc = self.E_diff_perc[ens]
        # ________________________________________________
        # Evapotranspiration
        self.potential_et_m_per_timestep = self.E_potential_et_m_per_timestep[ens]
        self.actual_et_m_per_timestep = self.E_actual_et_m_per_timestep[ens]
        # Added by Ryoko for soil-ode
        self.reduced_potential_et_m_per_timestep = self.E_reduced_potential_et_m_per_timestep[ens]
        self.actual_et_from_rain_m_per_timestep = self.E_actual_et_from_rain_m_per_timestep[ens]
        self.actual_et_from_soil_m_per_timestep = self.E_actual_et_from_soil_m_per_timestep[ens]
        # ________________________________________________________
        # Set these values now that we have the information from the configuration file.
        self.runoff_queue_m_per_timestep = self.E_runoff_queue_m_per_timestep[ens]
        self.num_giuh_ordinates = self.E_num_giuh_ordinates[ens]
        self.num_lateral_flow_nash_reservoirs = self.E_num_lateral_flow_nash_reservoirs[ens]
        self.nash_storage = self.E_nash_storage[ens]
        
        # ________________________________________________________
        # volume tracking variables.
        self.volstart = self.E_volstart[ens]
        self.vol_et_from_soil = self.E_vol_et_from_soil[ens]
        self.vol_et_from_rain = self.E_vol_et_from_rain[ens]
        self.vol_partition_runoff = self.E_vol_partition_runoff[ens]
        self.vol_partition_infilt = self.E_vol_partition_infilt[ens]
        self.vol_out_giuh = self.E_vol_out_giuh[ens]
        self.vol_end_giuh = self.E_vol_end_giuh[ens]
        self.vol_to_gw = self.E_vol_to_gw[ens]
        self.vol_to_gw_start = self.E_vol_to_gw_start[ens]
        self.vol_to_gw_end = self.E_vol_to_gw_end[ens]
        self.vol_from_gw = self.E_vol_from_gw[ens]
        self.vol_in_nash = self.E_vol_in_nash[ens]
        self.vol_in_nash_end = self.E_vol_in_nash_end[ens]
        self.vol_out_nash = self.E_vol_out_nash[ens]
        self.vol_soil_start = self.E_vol_soil_start[ens]
        self.vol_to_soil = self.E_vol_to_soil[ens]
        self.vol_soil_to_lat_flow = self.E_vol_soil_to_lat_flow[ens]
        self.vol_soil_to_gw = self.E_vol_soil_to_gw[ens]
        self.vol_soil_end = self.E_vol_soil_end[ens]
        self.volin = self.E_volin[ens]
        self.volout = self.E_volout[ens]
        self.volend = self.E_volend[ens]
        # Added by Ryoko for soil-ode
        self.vol_partition_runoff_IOF = self.E_vol_partition_runoff_IOF[ens]
        self.vol_partition_runoff_SOF = self.E_vol_partition_runoff_SOF[ens]
        self.vol_et_to_atm = self.E_vol_et_to_atm[ens]
        self.vol_et_from_soil = self.E_vol_et_from_soil[ens]
        self.vol_et_from_rain = self.E_vol_et_from_rain[ens]
        self.vol_PET = self.E_vol_PET[ens]
        self.soil_reservoir = self.E_soil_reservoir[ens]
        self.soil_reservoir_storage_deficit_m  = self.E_soil_reservoir_storage_deficit_m[ens]
        
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    def fill_ensemble_array_from_current_cfe_state(self):
        
        #----------------------------------#
        ens = self.current_ensemble_member #
        #----------------------------------#
        
        self.E_soil_reservoir[ens] = self.soil_reservoir
        self.E_gw_reservoir[ens] = self.gw_reservoir
        
        # ________________________________________________
        # Inputs
        self.E_timestep_rainfall_input_m[ens] = self.timestep_rainfall_input_m
        self.E_potential_et_m_per_s[ens] = self.potential_et_m_per_s
        
        # ________________________________________________
        # calculated flux variables
        # surface runoff that goes through the GIUH convolution process
        self.E_flux_overland_m[ens] = self.flux_overland_m
        # flux from soil to deeper groundwater reservoir
        self.E_flux_perc_m[ens] = self.flux_perc_m
        # lateral flux in the subsurface to the Nash cascade
        self.E_flux_lat_m[ens] = self.flux_lat_m
        # flux from the deep reservoir into the channels
        self.E_flux_from_deep_gw_to_chan_m[ens] = self.flux_from_deep_gw_to_chan_m
        # the available space in the conceptual groundwater reservoir
        self.E_gw_reservoir_storage_deficit_m[ens] = self.gw_reservoir_storage_deficit_m
        self.E_primary_flux[ens] = self.primary_flux
        self.E_secondary_flux[ens] = self.secondary_flux
        self.E_total_discharge[ens] = self.total_discharge
        # Added by Ryoko for soil-ode
        self.E_diff_infilt[ens] = self.diff_infilt
        self.E_diff_perc[ens] = self.diff_perc
        # ________________________________________________
        # Evapotranspiration
        self.E_potential_et_m_per_timestep[ens] = self.potential_et_m_per_timestep
        self.E_actual_et_m_per_timestep[ens] = self.actual_et_m_per_timestep
        # Added by Ryoko for soil-ode
        self.E_reduced_potential_et_m_per_timestep[ens] = self.reduced_potential_et_m_per_timestep
        self.E_actual_et_from_rain_m_per_timestep[ens] = self.actual_et_from_rain_m_per_timestep
        self.E_actual_et_from_soil_m_per_timestep[ens] = self.actual_et_from_soil_m_per_timestep
        # ________________________________________________________
        # Set these values now that we have the information from the configuration file.
        self.E_runoff_queue_m_per_timestep[ens] = self.runoff_queue_m_per_timestep
        self.E_num_giuh_ordinates[ens] = self.num_giuh_ordinates
        self.E_num_lateral_flow_nash_reservoirs[ens] = self.num_lateral_flow_nash_reservoirs
        self.E_nash_storage[ens] = self.nash_storage
        # ________________________________________________________
        # volume tracking variables.
        self.E_volstart[ens] = self.volstart
        self.E_vol_et_from_soil[ens] = self.vol_et_from_soil
        self.E_vol_et_from_rain[ens] = self.vol_et_from_rain
        self.E_vol_partition_runoff[ens] = self.vol_partition_runoff
        self.E_vol_partition_infilt[ens] = self.vol_partition_infilt
        self.E_vol_out_giuh[ens] = self.vol_out_giuh
        self.E_vol_end_giuh[ens] = self.vol_end_giuh = self.E_vol_end_giuh[ens]
        self.E_vol_to_gw[ens] = self.vol_to_gw
        self.E_vol_to_gw_start[ens] = self.vol_to_gw_start
        self.E_vol_to_gw_end[ens] = self.vol_to_gw_end
        self.E_vol_from_gw[ens] = self.vol_from_gw
        self.E_vol_in_nash[ens] = self.vol_in_nash
        self.E_vol_in_nash_end[ens] = self.vol_in_nash_end
        self.E_vol_out_nash[ens] = self.vol_out_nash
        self.E_vol_soil_start[ens] = self.vol_soil_start
        self.E_vol_to_soil[ens] = self.vol_to_soil
        self.E_vol_soil_to_lat_flow[ens] = self.vol_soil_to_lat_flow
        self.E_vol_soil_to_gw[ens] = self.vol_soil_to_gw
        self.E_vol_soil_end[ens] = self.vol_soil_end
        self.E_volin[ens] = self.volin
        self.E_volout[ens] = self.volout
        self.E_volend[ens] = self.volend
        # Added by Ryoko for soil-ode
        self.E_vol_partition_runoff_IOF[ens] = self.vol_partition_runoff_IOF
        self.E_vol_partition_runoff_SOF[ens] = self.vol_partition_runoff_SOF
        self.E_vol_et_to_atm[ens] = self.vol_et_to_atm
        self.E_vol_et_from_soil[ens] = self.vol_et_from_soil
        self.E_vol_et_from_rain[ens] = self.vol_et_from_rain
        self.E_vol_PET[ens] = self.vol_PET
        self.E_soil_reservoir[ens] = self.soil_reservoir
        self.E_soil_reservoir_storage_deficit_m[ens] = self.soil_reservoir_storage_deficit_m 