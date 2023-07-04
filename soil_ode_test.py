import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)

class soil_moisture_flux_ode(nn.Module):
    
    def __init__(self, cfe_state=None, reservoir=None):
        super(soil_moisture_flux_ode, self).__init__()
        self.cfe_state = cfe_state
        self.reservoir = reservoir

    def forward(self, t, states):
        
        S = states[0]
            
        storage_above_threshold_m = S - reservoir['storage_threshold_primary_m']
        storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
        storage_ratio = torch.minimum(storage_above_threshold_m / storage_diff, torch.tensor([1.0]))

        perc_lat_switch = torch.multiply(S - reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = torch.multiply(S - reservoir['wilting_point_m'] > 0, 1)

        storage_above_threshold_m_paw = S - reservoir['wilting_point_m']
        storage_diff_paw = reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']
        storage_ratio_paw = torch.minimum(storage_above_threshold_m_paw / storage_diff_paw, torch.tensor([1.0])) # Equation 11 (Ogden's document)
        dS_dt = cfe_state['infiltration_depth_m'] -1 * perc_lat_switch * (reservoir['coeff_primary'] + reservoir['coeff_secondary']) * storage_ratio - ET_switch * cfe_state['reduced_potential_et_m_per_timestep'] * storage_ratio_paw
        
        return (dS_dt)
    

# Initialization
y0 = torch.tensor([0.3])
t = torch.tensor([0, 0.05, 0.15, 0.3, 0.6, 1.0]) # ODE time descritization of one time step
cfe_state = {}
cfe_state['infiltration_depth_m'] = torch.tensor([0.1])
cfe_state['reduced_potential_et_m_per_timestep'] = torch.tensor([0.003])
reservoir = {}
reservoir['storage_threshold_primary_m'] = torch.tensor([0.2])
reservoir['storage_max_m'] = torch.tensor([0.4])
reservoir['wilting_point_m'] = torch.tensor([0.1])
reservoir['coeff_primary'] = torch.tensor([0.4])
reservoir['coeff_secondary'] = torch.tensor([0.4])

# Pass parameters beforehand
device = 'cpu'
func = soil_moisture_flux_ode(cfe_state=cfe_state, reservoir=reservoir).to(device)

# Solve and ODE
sol = odeint(
    func,
    y0,
    t,
    atol=1e-5,
    rtol=1e-5,
    # adjoint_params=()
)
