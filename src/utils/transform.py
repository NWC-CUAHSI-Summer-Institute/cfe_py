from sklearn import preprocessing
import torch
import logging
from omegaconf import DictConfig

# import utils.logger as logger
# from utils.read_yaml import config

log = logging.getLogger("utils.transform")
# log = logger.get_logger("utils.transform")


def to_physical(x, param, cfg: DictConfig):
    """
    The reverse scaling function to find the physical param from the scaled param (range [0,1))
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """
    range_ = cfg.transformation[param]
    x_ = x * (range_[1] - range_[0])
    output = x_ + range_[0]
    return output


def from_physical(x, param):
    """
    The scaling function to convert a physical param to a value within [0,1)
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """


def normalization(x):
    """
    A min/max Scaler for each feature to be fed into the MLP
    :param x:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_trans = x.transpose(2, 0).transpose(2, 1)
    x_tensor = torch.zeros(x_trans.shape)

    # Apply the scaling to each attribute across multiple basins
    # therefore the max value is the maximum of an attribute for all the target basins
    for i in range(0, x_trans.shape[0]):
        x_tensor[i, :] = torch.tensor(
            min_max_scaler.fit_transform(x_trans[i, :].transpose(1, 0))
        ).transpose(1, 0)
    """Transposing to do correct normalization"""
    return x_tensor.transpose(1, 0).transpose(1, 2)
