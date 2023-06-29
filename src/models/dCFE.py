from omegaconf import DictConfig
import logging
import time
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

log = logging.getLogger("models.dCFE")


class dCFE(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        """
        super(dCFE, self).__init__()

        self.cfg = cfg

        # Setting NN parameters
        self.slope = nn.Parameter(torch.tensor(0.0))

    def forward(self, x) -> (Tensor, Tensor):
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
        # TODO implement the CFE functinos
        runoff = 0
        return runoff
