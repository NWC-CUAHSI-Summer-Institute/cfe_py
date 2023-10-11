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
        input_size = self.cfg.models.mlp.num_attrs * len(Data)
        hidden_size = self.cfg.models.mlp.hidden_size
        output_size = self.cfg.models.mlp.num_params * len(
            Data
        )  # self.cfg.models.mlp.output_size

        self.m1 = Flatten(start_dim=0, end_dim=1)
        self.m2 = Unflatten(
            dim=0, unflattened_size=(self.cfg.models.mlp.num_params, len(Data))
        )

        # Defining the layers using nn.Sequential
        self.network = nn.Sequential(
            Linear(input_size, hidden_size),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, output_size),
            Sigmoid(),
        )

        """
        self.m1 = Flatten(start_dim=0, end_dim=1)
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, hidden_size)
        self.lin3 = Linear(hidden_size, hidden_size)
        self.lin4 = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        """
        # self.ReLu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # 4/? If you look at the end of MLP.forward() you'll see that
        # I'm transforming the outputs of the NN into parameter space.
        # Here is the code to do that

        # x: attribute Tensor, [[P],[PET]] (2, timestep)
        # is Flattened to [[P,PET]] (1, timestep*2)
        _x = self.m1(x.transpose(0, 1))
        _out1 = self.network(_x.transpose(0, 1))
        # x1 = self.lin1(_x)
        # x2 = self.lin2(x1)
        # x3 = self.lin3(x2)
        # x4 = self.lin4(x3)
        # _out1 = self.sigmoid(x4)
        # Possibly, HardTanh? https://paperswithcode.com/method/hardtanh-activation
        out1 = self.m2(_out1)
        # x_transpose = out1.transpose(0, 1) # No transpose though ...

        refkdt = to_physical(x=out1[0], param="refkdt", cfg=self.cfg.models)
        satdk = to_physical(x=out1[1], param="satdk", cfg=self.cfg.models)
        return refkdt, satdk
