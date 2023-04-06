# Python version of the NWM Conceptual Functional Equivalent (CFE) model

Note this version is for prototyping, research and development.  
The **official** CFE code lives here: [https://github.com/NOAA-OWP/cfe/](https://github.com/NOAA-OWP/cfe/)

# Files
## environment.yml: 
This is an environment file with the required Python libraries needed to run the model with BMI. Create the environment with this command: `conda env create -f environment.yml`, then activate it with `conda activate bmi_cfe` 
## cfe.py
This is the main model code. The input to this model is a CFE State, which can be either a Python class, or a dictionary. The only requirement is that the object contains the entire model running state (Not just state variables), which concists of the following:  
 - Forcings
    - timestep_rainfall_input_m
    - potential_et_m_per_s
 - Parameters
    - soil_params
    - K_nash
    - etc.
 - Volume trackers
    - volin
    - vol_to_gw
    - volout
    - etc.
 - Fluxes
    - flux_Qout_m
    - flux_giuh_runoff_m
    - flux_nash_lateral_runoff_m
    - flux_from_deep_gw_to_chan_m
    - total_discharge
 - Etc.  
 The model code takes the running state and calculates all the fluxes and corresponding state changes. The running state contains the single timestep changes, nothing is "returned" from the function. All the processing and interpretation of the model should take place in the driving code. An example of driving code is below.
 ## bmi_cfe.py
This is the code for the Basic Model Interface (BMI) that is used to call the cfe and interact with other models via the Framework, or driving code. This code contains all the required BMI functions to run the CFE, including 
 - initialize: Perform startup tasks for the model.
 - update: Advance model state by one time step. Calls the function `run_cfe` from `cfe.py`
 - update_until: Advance model state until the given time.
 - finalize: Perform tear-down tasks for the model.
 - get_value: Get a copy of values of a given variable.
 - set_value: Set the values of a given variable.
 - etc.  
 These functions need to be called by a framework or driving code, an example of which is below.
 ## run_cfe_bmi.ipynb
 This is an example run for the CFE. The Jupyter notebook is good for visualizing the results. Notice that there are blocks of code that call all the functions listed above. These are the main BMI functions that allow us to control and run the model. This example requires a configuration file, which BMI uses to set the specifics of the model, including how to use Forcings. More on the configuration file below.
 ## cat58_config_cfe.json
 This file has all the information to configure the model for a specific basin. The forcing file can be specified to run the a comparison with the origional model code, and there should be a corresponding file with the output from the test (compare_results_file). In general the model should be run getting forcing from the driver using the set_value function. Some of the values in the config file will come from the NWM parameters, and some will be calibrated. Some values are basin specific, and need to be set to get the correct results for the basin, for instance the catchment_area_km2 is needed to convert the runoff to a volume flux, rather than a depth.


## Parameters 
Copy and pasted from official repo. To be edited. 

| Variable | Datatype |  Limits  | Units | Role | Process | Description |
| -------- | -------- | ------ | ----- | ---- | ------- | ----------- |
| forcing_file | *char* | 256  |   | filename |   | path to forcing inputs csv; set to `BMI` if passed via `bmi.set_value*()`  |
| soil_params.depth | *double* |   | meters [m]| state |  | soil depth  |
| soil_params.b | *double* |   |   | state |   | beta exponent on Clapp-Hornberger (1978) soil water relations  |
| soil_params.satdk | *double* |   |  meters/second [m s-1] | state |  | saturated hydraulic conductivity  |
| soil_params.satpsi  | *double* |   |  meters [m] | state |  | saturated capillary head  |
| soil_params.slop   | *double* |   |  meters/meters [m/m]| state |  | this factor (0-1) modifies the gradient of the hydraulic head at the soil bottom.  0=no-flow. |
| soil_params.smcmax  | *double* |   |  meters/meters [m/m] | state |  | saturated soil moisture content  |
| soil_params.wltsmc | *double* |   |  meters/meters [m/m] | state |   | wilting point soil moisture content  |
| soil_params.expon  | *double* |   |  | parameter_adjustable |    | optional; defaults to `1.0`  |
| soil_params.expon_secondary  | *double* |  |   | parameter_adjustable |  | optional; defaults to `1.0` |
| max_gw_storage | *double* |   |  meters [m] | parameter_adjustable |  | maximum storage in the conceptual reservoir |
| Cgw | *double* |   |  meters/hour [m h-1] | parameter_adjustable |  | the primary outlet coefficient |
| expon | *double* |   |   | parameter_adjustable |  | exponent parameter (1.0 for linear reservoir) |
| gw_storage | *double* |   |  meters/meters [m/m] | parameter_adjustable |  | initial condition for groundwater reservoir - it is the ground water as a decimal fraction of the maximum groundwater storage (max_gw_storage) for the initial timestep |
| alpha_fc | *double* |   |   | parameter_adjustable |  | field capacity |
| soil_storage| *double* |   | meters/meters [m/m] | parameter_adjustable |  | initial condition for soil reservoir - it is the water in the soil as a decimal fraction of maximum soil water storage (smcmax * depth) for the initial timestep |
| K_nash | *int* |   |   | parameter_adjustable |   | number of Nash lf reservoirs (optional, defaults to 2, ignored if storage values present)  |
| K_lf | *double* |   |   | parameter_adjustable |  | Nash Config param - primary reservoir  |
| nash_storage | *double* |   |   | parameter_adjustable |  | Nash Config param - secondary reservoir   |
| giuh_ordinates   | *double* |   |   | parameter_adjustable |  | Giuh ordinates in dt time steps   |
| num_timesteps  | *int* |   |  | time_info |  | set to `1` if `forcing_file=BMI`   |
| verbosity | *int* | `0`-`3`  |   | option |   |  prints various debug and bmi info  |
| surface_partitioning_scheme | *char* | `Xinanjiang` or `Schaake`  |  | parameter_adjustable | direct runoff |    |
| a_Xinanjiang_inflection_point_parameter | *double* |   |  | parameter_adjustable | direct runoff | when `surface_partitioning_scheme=Xinanjiang`   |
| b_Xinanjiang_shape_parameter=1  | *double* |   |   | parameter_adjustable  | direct runoff | when `surface_partitioning_scheme=Xinanjiang`   |
| x_Xinanjiang_shape_parameter=1  | *double* |   |   | parameter_adjustable | direct runoff | when `surface_partitioning_scheme=Xinanjiang`   |
| aet_rootzone                    | *boolean* | True, true or 1  |  | coupling parameter | `rootzone-based AET` | when `CFE coupled to SoilMoistureProfile` |
| max_root_zone_layer | *double* |  | meters [m] | parameter_adjustable | AET | layer of the soil that is the maximum root zone depth. That is, the depth of the layer where the AET is drawn from |
| soil_layer_depths | 1D array |  | meters [m] | parameter_adjustable | AET | an array of depths from the surface. Example, soil_layer_depths=0.1,0.4,1.0,2.0
| sft_coupled                     | *boolean* | True, true or 1  |  | coupling parameter | `ice-fraction based runoff` | when `CFE coupled to SoilFreezeThaw`|