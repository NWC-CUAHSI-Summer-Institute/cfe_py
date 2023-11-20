import logging
from omegaconf import DictConfig
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import hydroeval as he
import os
import pandas as pd

import torch

torch.set_default_dtype(torch.float64)
# torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from agents.base import BaseAgent
from data.Data import Data
from data.metrics import calculate_nse
from models.dCFE import dCFE
from utils.ddp_setup import find_free_port, cleanup


log = logging.getLogger("agents.DifferentiableCFE")

# Set the RANK environment variable manually

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

        self.cfg = cfg
        # Setting the cfg object and manual seed for reproducibility
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)

        # Defining the torch Dataset and Dataloader
        self.data = Data(self.cfg)
        self.data_loader = DataLoader(self.data, batch_size=1, shuffle=True)

        # Defining the model
        self.model = dCFE(cfg=self.cfg, Data=self.data)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.models.hyperparameters.learning_rate
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=cfg.models.hyperparameters.step_size,
            gamma=cfg.models.hyperparameters.gamma,
        )

        self.current_epoch = 0

        self.output_dir = self.create_output_dir()

        self.states = torch.zeros(
            [
                self.data.num_basins,
                self.data.n_timesteps,
                self.cfg.models.mlp.num_states,
            ]
        )

        # # Prepare for the DDP
        # free_port = find_free_port()
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = free_port

    def create_output_dir(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        dir_name = f"{current_date}_output"
        output_dir = os.path.join(self.cfg.output_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            log.info("You have entered CTRL+C.. Wait to finalize")

    def train(self) -> None:
        """
        Execute the training process.

        Sets the model to train mode, sets up DistributedDataParallel, and initiates training for the number of epochs
        specified in the configuration.
        """
        self.model.train()  # this .train() is a function from nn.Module

        self.loss_record = np.zeros(self.cfg.models.hyperparameters.epochs)

        # dist.init_process_group(
        #     backend="gloo",
        #     world_size=0,
        #     rank=self.cfg.num_processes,
        # )

        # Create the DDP object with the GLOO backend
        # self.net = DDP(self.model.to(self.cfg.device), device_ids=None)

        for epoch in range(1, self.cfg.models.hyperparameters.epochs + 1):
            log.info(f"Epoch #: {epoch}/{self.cfg.models.hyperparameters.epochs}")
            self.loss_record[epoch - 1] = self.train_one_epoch()
            # print("Start mlp forward")
            # self.model.mlp_forward()
            # print("End mlp forward")
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        # Reset
        self.optimizer.zero_grad()
        self.model.cfe_instance.reset_volume_tracking()

        # Reset the model states and parameters
        # refkdt and satdk gets updated in the model as well
        self.model.mlp_forward(self.states)
        self.model.initialize()

        y_hat = torch.empty(
            [self.data.num_basins, self.data.n_timesteps], device=self.cfg.device
        )
        y_hat.fill_(float("nan"))

        # y_hat = torch.zeros(n, device=self.cfg.device)  # runoff

        for t, (x, y_t) in enumerate(tqdm(self.data_loader, desc="Processing data")):
            runoff, cfe_states = self.model(x, t)  #
            y_hat[:, t] = runoff
            self.states[:, t, :] = cfe_states.detach()

        # Run the following to get a visual image of tesnors
        #######
        # from torchviz import make_dot
        # a = make_dot(loss, params=self.model.c)
        # a.render("backward_computation_graph")
        #######

        loss = self.validate(y_hat, self.data.y)

        return loss

        # From https://github.com/mhpi/differentiable_routing/blob/26dd83852a6ee4094bd9821b2461a7f528efea96/src/agents/graph_network.py
        #######
        # with open(
        #     f"{self.PATH}GNN_output_steps-{config['time']['steps']}_{self.save_name}_{self.rank}_epoch-{self.current_epoch}.npy",
        #     "wb",
        # ) as f:
        #     np.save(f, y_hat.detach().numpy())
        #######

    def validate(self, y_hat_: Tensor, y_t_: Tensor) -> None:
        """
        One cycle of model validation
        This function calculates the loss for the given predicted and actual values,
        backpropagates the error, and updates the model parameters.

        Parameters:
        - y_hat_ : The tensor containing predicted values
        - y_t_ : The tensor containing actual values.
        """

        # Transform validation/output data for validation
        y_t_ = y_t_.squeeze()
        warmup = self.cfg.models.hyperparameters.warmup
        y_hat = y_hat_[:, warmup:]
        y_t = y_t_[:, warmup:]

        y_hat_np = y_hat_.detach().numpy()
        y_t_np = y_t_.detach().numpy()

        # Save results
        # Evaluate
        kge = he.evaluator(he.kge, y_hat_np[0], y_t_np[0])
        log.info(
            f"trained KGE for the basin {self.data.basin_ids[0]}: {float(kge[0]):.4}"
        )

        self.save_result(
            y_hat=y_hat_np,
            y_t=y_t_np,
            out_filename=f"epoch{self.current_epoch}",
            plot_figure=False,
        )

        # Compute the overall loss
        mask = torch.isnan(y_t)
        y_t_dropped = y_t[~mask]
        y_hat_dropped = y_hat[~mask]
        if y_hat_dropped.shape != y_t_dropped.shape:
            print("y_t and y_hat shape not matching")

        print("calculate loss")
        loss = self.criterion(y_hat_dropped, y_t_dropped)
        log.info(f"loss at epoch {self.current_epoch}: {loss:.6f}")

        # Backpropagate the error
        start = time.perf_counter()
        print("Loss backward starts")
        loss.backward()
        print("Loss backward ends")
        end = time.perf_counter()

        # Log the time taken for backpropagation and the calculated loss
        log.debug(f"Back prop took : {(end - start):.6f} seconds")
        log.debug(f"Loss: {loss}")

        # Save results
        # TODO: add to save loss

        # Update the model parameters
        self.model.print()
        print("Start optimizer")
        self.optimizer.step()
        self.scheduler.step()
        print("End optimizer")

        return loss

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
            y_hat = torch.zeros_like(self.data.y, device=self.cfg.device)

            # Run one last time
            for t, (x, y_t) in enumerate(
                tqdm(self.data_loader, desc="Processing data")
            ):
                runoff = self.model(x, t)  #
                y_hat[:, t] = runoff.transpose(dim0=0, dim1=1)

            y_hat_ = y_hat.detach().numpy()
            y_t_ = self.data.y.detach().numpy()

            self.save_result(
                y_hat=y_hat_,
                y_t=y_t_,
                out_filename="final_result",
                plot_figure=True,
            )

            print(self.model.finalize())

        except:
            raise NotImplementedError

        # Save the loss
        self.save_loss()

    def save_loss(self):
        df = pd.DataFrame(self.loss_record)
        file_path = os.path.join(self.output_dir, f"final_result_loss.csv")
        df.to_csv(file_path)

        fig, axes = plt.subplots()
        axes.plot(self.loss_record, "-")
        axes.set_title(
            f"Learning rate: {self.cfg.models.hyperparameters.learning_rate}"
        )
        axes.set_ylabel("loss")
        axes.set_xlabel("epoch")
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"final_result_loss.png"))
        plt.close()

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

    def save_result(self, y_hat, y_t, out_filename, plot_figure=False):
        # Save all basin runs

        refkdt_ = self.model.refkdt.detach().numpy()
        satdk_ = self.model.satdk.detach().numpy()

        warmup = self.cfg.models.hyperparameters.warmup

        for i, basin_id in enumerate(self.data.basin_ids):
            # Save the timeseries of runoff and the best dynamic parametersers

            data = {
                "refkdt": refkdt_[i, warmup:],
                "satdk": satdk_[i, warmup:],
                "y_hat": y_hat[i, warmup:].squeeze(),
                "y_t": y_t[i, warmup:].squeeze(),
            }
            df = pd.DataFrame(data)
            df.to_csv(
                os.path.join(self.output_dir, f"{out_filename}_{basin_id}.csv"),
                index=False,
            )

            if plot_figure:
                # Plot
                eval_metrics = he.evaluator(he.kge, y_hat[i], y_t[i])[0]
                fig, axes = plt.subplots(figsize=(5, 5))
                axes.plot(y_t[i, warmup:], "-", label="eval (synthetic)", alpha=0.5)
                axes.plot(y_hat[i, warmup:], "--", label="sim (recovery)", alpha=0.5)
                axes.set_title(f"Classic (KGE={float(eval_metrics):.2})")
                plt.legend()
                plt.savefig(
                    os.path.join(self.output_dir, f"{out_filename}_{basin_id}.png")
                )
                plt.close()
