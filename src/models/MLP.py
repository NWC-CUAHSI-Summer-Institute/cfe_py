import torch
import math

# from torch.nn import Linear, Sigmoid
from omegaconf import DictConfig
from torch.nn import Sigmoid, Linear
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
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # 3/? here is the MLP code.
        # This uses a Sigmoid activation function.
        # ReLU may be more effective in your case:
        torch.manual_seed(0)
        input_size = self.cfg["src\models"].mlp.input_size
        hidden_size = self.cfg["src\models"].mlp.hidden_size
        output_size = self.cfg["src\models"].mlp.output_size
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, hidden_size)
        self.lin3 = Linear(hidden_size, hidden_size)
        self.lin4 = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # 4/? If you look at the end of MLP.forward() you'll see that
        # I'm transforming the outputs of the NN into parameter space.
        # Here is the code to do that

        x1 = self.lin1(x)
        x2 = self.lin2(x1)
        x3 = self.lin3(x2)
        x4 = self.lin4(x3)
        out1 = self.sigmoid(x4)
        x_transpose = out1.transpose(0, 1)
        refkdt = to_physical(x_transpose[0], "refkdt")
        satdk = to_physical(x_transpose[1], "satdk")
        return refkdt, satdk
