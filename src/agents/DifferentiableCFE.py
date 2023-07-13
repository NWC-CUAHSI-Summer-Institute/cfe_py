import logging
from omegaconf import DictConfig
import time
import torch

# torch.autograd.set_detect_anomaly(True)

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

import json

log = logging.getLogger("agents.DifferentiableLGAR")

# Refer to https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py#L98
# self.model is https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/graph/models/GNN_baseline.py#L25


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
        self.model = dCFE(cfg=self.cfg, Data=self.data)

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

    # Previous train function before the NN
    # def train(self):
    #     """
    #     Main training loop
    #     :return:
    #     """
    #     self.model.train()
    #     for epoch in range(1, self.cfg["src\models"].hyperparameters.epochs + 1):
    #         self.train_one_epoch()
    #         self.current_epoch += 1

    def train(self) -> None:
        """
        Execute the training process.

        Sets the model to train mode, sets up DistributedDataParallel, and initiates training for the number of epochs
        specified in the configuration.
        """
        self.model.train()  # .train() is a function from nn.Module
        self.net = None  # Just for CPU
        # self.net = DDP(
        #     self.model, device_ids=None
        # )  # Device IDS are only used on the GPU
        self.model.mlp_forward()
        for epoch in range(1, self.cfg["src\models"].hyperparameters.epochs + 1):
            self.data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch()
            self.model.mlp_forward()
            self.plot()
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.optimizer.zero_grad()
        self.model.cfe_instance.reset_volume_tracking()
        self.model.cfe_instance.reset_flux_and_states()

        n = self.data.n_timesteps
        y_hat = torch.zeros([n], device=self.cfg.device)  # runoff

        for i, (x, y_t) in enumerate(tqdm(self.data_loader, desc="Processing data")):
            runoff = self.model(x)
            y_hat[i] = runoff

        # Run the following to get a visual image of tesnors
        # from torchviz import make_dot
        # a = make_dot(loss, params=self.model.c)
        # a.render("backward_computation_graph")

        self.validate(y_hat, self.data.y)

        # From https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py
        # with open(
        #     f"{self.PATH}GNN_output_steps-{config['time']['steps']}_{self.save_name}_{self.rank}_epoch-{self.current_epoch}.npy",
        #     "wb",
        # ) as f:
        #     np.save(f, y_hat.detach().numpy())

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

        y_hat_np = y_hat_.detach().numpy()
        y_t_np = y_t_.detach().numpy()

        kge = he.evaluator(he.kge, y_hat.detach().numpy(), y_t.detach().numpy())
        log.info(f"trained KGE: {float(kge[0]):.4}")

        self.save_result(
            y_hat=y_hat_np,
            y_t=y_t_np,
            eval_metrics=kge[0],
            out_filename="test_ts_before_backward_propagation",
        )

        # Compute the overall loss
        mask = torch.isnan(y_t)
        y_t_dropped = y_t[~mask]
        y_hat_dropped = y_hat[~mask]
        loss = self.criterion(y_hat_dropped, y_t_dropped)

        # Backpropagate the error
        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()

        # Log the time taken for backpropagation and the calculated loss
        log.debug(f"Back prop took : {(end - start):.6f} seconds")
        log.debug(f"Loss: {loss}")

        # Update the model parameters
        self.optimizer.step()

        self.model.print()

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """

        try:
            # Get the final training
            self.model.cfe_instance.reset_volume_tracking()
            self.model.cfe_instance.reset_flux_and_states()
            n = self.data.n_timesteps
            y_hat = torch.zeros([n], device=self.cfg.device)  # runoff

            for i, (x, y_t) in enumerate(
                tqdm(self.data_loader, desc="Processing data")
            ):
                runoff = self.model(x)
                y_hat[i] = runoff

            y_hat_ = y_hat.detach().numpy()
            y_t_ = self.data.y.detach().numpy()

            kge = he.evaluator(he.kge, y_hat_, y_t_)

            self.save_result(
                y_hat=y_hat_,
                y_t=y_t_,
                eval_metrics=kge[0],
                out_filename="test_ts_after_backward_propagation",
            )

            print(self.model.finalize())

        except:
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

    def save_result(self, y_hat, y_t, eval_metrics, out_filename):
        # Get the folder
        folder_pattern = rf".\output\{datetime.now():%Y-%m-%d}_*"
        matching_folder = glob.glob(folder_pattern)

        # Timeseries of runoff
        np.savetxt(
            os.path.join(matching_folder[0], f"{out_filename}.csv"),
            np.stack([y_hat, y_t]).transpose(),
            delimiter=",",
        )

        # Plot
        fig, axes = plt.subplots(figsize=(5, 5))
        axes.plot(y_t, label="observed")
        axes.plot(y_hat, label="simulated")
        axes.set_title(f"Classic (KGE={float(eval_metrics):.4})")
        plt.legend()
        plt.savefig(os.path.join(matching_folder[0], f"{out_filename}.png"))

        # Best param
        array_dict = {
            key: tensor.detach().numpy().tolist()
            for key, tensor in self.model.c.items()
        }
        with open(
            os.path.join(matching_folder[0], "best_params.json"), "w"
        ) as json_file:
            json.dump(array_dict, json_file, indent=4)
