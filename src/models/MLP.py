"""A class to hold the Multilayer Perceptron Model to estimate soil parameters"""
import torch
import math

torch.set_default_dtype(torch.float64)
# from torch.nn import Linear, Sigmoid
from omegaconf import DictConfig
from torch.nn import Flatten, Unflatten, Sigmoid, Linear, ReLU
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys

# import utils.logger as logger
# from utils.read_yaml import config
from utils.transform import to_physical

# log = logger.get_logger("graphs.MLP")


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig, Data) -> None:
        super().__init__()
        """
        The Multilayer Perceptron Model (MLP) which learns values
        of refkdt and satdk from downstream discharge

        args:
        - cfg: The DictConfig object that houses global variables
        - Data: The input data tensor
        """

        self.cfg = cfg

        # The size of the attributes going into MLP corresponds to self.cfg.models.mlp.input_size
        # The size of out1 from MLP correponds to self.cfg.models.mlp.output_size (sso increase this when increasing parameters)

        torch.manual_seed(0)
        input_size = Data.num_basins * len(Data) * self.cfg.models.mlp.num_attrs
        hidden_size = self.cfg.models.mlp.hidden_size
        output_size = (
            Data.num_basins * len(Data) * self.cfg.models.mlp.num_params
        )  # self.cfg.models.mlp.output_size
        output_shape = (self.cfg.models.mlp.num_params, len(Data), Data.num_basins)

        self.m1 = Flatten(start_dim=0, end_dim=-1)
        self.m2 = Unflatten(dim=0, unflattened_size=output_shape)

        # Defining the layers using nn.Sequential
        self.network = nn.Sequential(
            Linear(input_size, hidden_size),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, output_size),
            Sigmoid(),
        )

        # Possibly, HardTanh or ReLU()? https://paperswithcode.com/method/hardtanh-activation

    def forward(self, x: Tensor) -> Tensor:
        # x.transpose(0, -1): attribute tensor [[[P],[PET]]] (num_attributes(3), timestep, num_basin)
        # is Flattened to [[P],[PET]] (timestep*num_basins*num_attributes(3))
        _x = self.m1(x.transpose(0, -1))
        _out1 = self.network(_x)

        # _out1 (timestep*num_basins*num_parameters(2))
        # is Unflattened to (num_parameters(3), timestep, num_basin)
        out1 = self.m2(_out1)
        # x_transpose is (num_basin, timestep, num_parameters(3))
        x_transpose = out1.transpose(0, -1)

        # transforming the outputs of the NN into parameter space
        # (num_basin, timestep)
        refkdt = to_physical(
            x=x_transpose[:, :, 0], param="refkdt", cfg=self.cfg.models
        )
        satdk = to_physical(x=x_transpose[:, :, 1], param="satdk", cfg=self.cfg.models)
        return refkdt, satdk
