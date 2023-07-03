import logging
from omegaconf import DictConfig
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.agents.base import BaseAgent
from src.data.Data import Data
from src.data.metrics import calculate_nse
from src.models.dCFE import dCFE

import numpy as np

import hydroeval as he

import matplotlib.pyplot as plt
from datetime import datetime

import glob
import os

log = logging.getLogger("agents.DifferentiableLGAR")


class DifferentiableCFE(BaseAgent):
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
        self.model = dCFE(self.cfg)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["src\models"].hyperparameters.learning_rate
        )

        self.current_epoch = 0

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            log.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        self.model.train()
        for epoch in range(1, self.cfg["src\models"].hyperparameters.epochs + 1):
            self.train_one_epoch()
            self.current_epoch += 1
        

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.optimizer.zero_grad()

        n = self.data.n_timesteps
        # y_hat = np.zeros([n])
        y_hat = torch.zeros([n], device=self.cfg.device)  # runoff

        for i, (x, y_t) in enumerate(tqdm(self.data_loader, desc="Processing data")):
            runoff = self.model(x)
            y_hat[i] = runoff
            
        self.validate(y_hat, self.data.y)
        
    def validate(self, y_hat_: Tensor, y_t_: Tensor) -> None:
        """
        One cycle of model validation
        This function calculates the loss for the given predicted and actual values,
        backpropagates the error, and updates the model parameters.

        Parameters:
        - y_hat_ : The tensor containing predicted values
        - y_t_ : The tensor containing actual values.
        """
        warmup = self.cfg["src\models"].hyperparameters.warmup
        y_hat = y_hat_[warmup:]
        y_t = y_t_[warmup:]

        # Outputting trained KGE coefficient
        """Numpy implementation
        kge = he.evaluator(he.kge, simulations=y_hat, evaluation=y_t)
        log.info(
            f"trained KGE: {float(kge[0]):.4}"
        )
        np.savetxt(r'.\output\testrun.csv', np.stack([y_hat, y_t]).transpose(), delimiter=',')
        """
        
        kge = he.evaluator(he.kge, y_hat.detach().numpy(), y_t.detach().numpy())
        log.info(
            f"trained KGE: {float(kge[0]):.4}"
        )

        # Compute the overall loss
        loss = self.criterion(y_hat, y_t)

        # Backpropagate the error
        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()

        # Log the time taken for backpropagation and the calculated loss
        log.debug(f"Back prop took : {(end - start):.6f} seconds")
        log.debug(f"Loss: {loss}")

        # Update the model parameters
        self.optimizer.step()

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # Get hte final training
        n = self.data.n_timesteps
        y_hat = torch.zeros([n], device=self.cfg.device)  # runoff

        for i, (x, y_t) in enumerate(tqdm(self.data_loader, desc="Processing data")):
            runoff = self.model(x)
            y_hat[i] = runoff
            
        y_hat_ = y_hat.detach().numpy()
        y_t_ = self.data.y.detach().numpy()
            
        kge = he.evaluator(he.kge, y_hat_, y_t_)
        
        # Save the results
        # Define the pattern for the folder name
        folder_pattern = fr".\output\{datetime.now():%Y-%m-%d}_*"
        matching_folders = glob.glob(folder_pattern)
        np.savetxt(os.path.join(matching_folders[-1], 'test.csv'), np.stack([y_hat_, y_t_]).transpose(), delimiter=',')

        fig, axes = plt.subplots(figsize=(5, 5))       
        axes.plot(y_t_, label='observed')
        axes.plot(y_hat_, label='simulated')
        axes.set_title(f'ODE (KGE={float(kge[0]):.4})')
        plt.legend()
        plt.savefig(os.path.join(matching_folders[-1], 'test.png'))
                
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
