import logging
from omegaconf import DictConfig
import time
import torch

torch.set_default_dtype(torch.float64)
from pathlib import Path

# torch.autograd.set_detect_anomaly(True)

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from agents.base import BaseAgent
from data.Data import Data
from data.metrics import calculate_nse
from models.dCFE import dCFE
from models.SyntheticCFE import SyntheticCFE
from utils.ddp_setup import find_free_port, cleanup

import numpy as np
import pandas as pd

import hydroeval as he

import matplotlib.pyplot as plt
from datetime import datetime

import glob
import os

import json

log = logging.getLogger("agents.DifferentiableLGAR")


class SyntheticAgent(BaseAgent):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Differentiable LGAR code

        Sets up the initial state of the agent
        :param cfg:
        """
        super().__init__()

        # Setting the cfg object and manual seed for reproducibility
        self.cfg = cfg
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)

        # Defining the torch Dataset and Dataloader
        self.data = Data(self.cfg)
        self.data_loader = DataLoader(self.data, batch_size=1, shuffle=False)

        # Defining the model and output variables to save
        self.model = SyntheticCFE(cfg=self.cfg, Data=self.data)

    def run(self):
        # Reset the model states and parameters
        # refkdt and satdk gets updated in the model as well
        # Initialize in dCFE?
        # self.model.cfe_instance.refkdt = self.model.refkdt  # .squeeze(dim=0)
        # self.model.cfe_instance.satdk = self.model.satdk  # .squeeze(dim=0)
        # self.model.cfe_instance.reset_flux_and_states()
        try:
            n = self.data.n_timesteps
            y_hat = torch.zeros_like(self.data.y, device=self.cfg.device)  # runoff
            outputs = self.model.cfe_instance.get_output_var_names()
            output_lists = {output: [] for output in outputs}

            with torch.no_grad():
                for i, (x, y_t) in enumerate(
                    tqdm(self.data_loader, desc="Processing data")
                ):
                    runoff = self.model(x)
                    y_hat[:, i, :] = runoff.T
                    for output in outputs:
                        output_lists[output].append(
                            self.model.cfe_instance.get_value(output)
                        )

            self.save_data(y_hat)
            self.save_other_fluxes(output_lists)
            self.model.print()

        except KeyboardInterrupt:
            interrupt = True
            self.finalize(interrupt)
            log.info("You have entered CTRL+C.. Wait to finalize")

    def save_data(self, y_hat_: Tensor) -> None:
        """
        One cycle of model validation
        This function calculates the loss for the given predicted and actual values,
        backpropagates the error, and updates the model parameters.

        Parameters:
        - y_hat_ : The tensor containing predicted values
        - y_t_ : The tensor containing actual values.
        """

        y_hat_np = y_hat_.numpy()

        date_range = pd.date_range(
            start=self.data.start_time, end=self.data.end_time, freq="H"
        )

        # Creating column names dynamically
        y_hat_df = pd.DataFrame(
            y_hat_np[:, :, 0].T, index=date_range, columns=self.cfg.data.basin_ids
        )

        # Define the output directory
        dir_path = Path(self.cfg.synthetic.output_dir)
        # Check if the directory exists, if not, create it
        dir_path.mkdir(parents=True, exist_ok=True)

        # Define the output file path
        file_path = dir_path / f"synthetic_{self.cfg.soil_scheme}.csv"

        # Save the numpy array to the file
        y_hat_df.to_csv(file_path)

    def save_other_fluxes(self, output_lists):
        for output, values in output_lists.items():
            # Convert the list of values to a pandas DataFrame
            df = pd.DataFrame(
                np.array(values).reshape(-1, 3), columns=self.data.basin_ids
            )

            # Save the DataFrame to a CSV file
            dir_path = Path(self.cfg.synthetic.output_dir)
            file_path = dir_path / f"{output}.csv"
            df.to_csv(file_path, index=False)

    def train(self):
        try:
            print("finished the run")
        except:
            raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def finalize(self, interrupt=False):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        try:
            self.model.cfe_instance.finalize(print_mass_balance=True)
            print(f"Agend finished the job")
        except:
            raise NotImplementedError
