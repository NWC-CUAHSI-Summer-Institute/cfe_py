import logging
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dpLGAR.agents.base import BaseAgent
from dpLGAR.data.Data import Data
from dpLGAR.data.metrics import calculate_nse
from dpLGAR.models.SyntheticLGAR import SyntheticLGAR
from dpLGAR.models.physics.MassBalance import MassBalance

log = logging.getLogger("agents.SyntheticAgent")


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

        # Initialize DistributedDataParallel (DDP)
        self.rank = int(cfg.local_rank)

        # Configuring subcycles (If we're running hourly, 15 mins, etc)
        self.cfg.models.subcycle_length_h = self.cfg.models.subcycle_length * (
            1 / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.forcing_resolution_h = (
            self.cfg.models.forcing_resolution / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.num_subcycles = int(
            self.cfg.models.forcing_resolution_h / self.cfg.models.subcycle_length_h
        )

        # Setting the number of values per batch (Currently we want this to be 1)
        self.hourly_mini_batch = int(cfg.models.hyperparameters.minibatch * 24)

        # Defining the torch Dataset and Dataloader
        self.data = Data(self.cfg)
        self.data_loader = DataLoader(
            self.data,
            batch_size=self.hourly_mini_batch,
            shuffle=False,
        )

        # Defining the model and output variables to save
        self.model = SyntheticLGAR(self.cfg)
        self.mass_balance = MassBalance(cfg, self.model)

        self.current_epoch = 0

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.model.eval()
            y_hat_ = torch.zeros([len(self.data_loader)], device=self.cfg.device)  # runoff
            y_t_ = torch.zeros([len(self.data_loader)], device=self.cfg.device)  # runoff
            with torch.no_grad:
                for i, (x, y_t) in enumerate(
                        tqdm(
                            self.data_loader,
                            desc=f"Nproc: {self.rank} Epoch {self.current_epoch + 1} Training",
                        )
                ):
                    # Resetting output vars
                    runoff = self.model(i, x.squeeze())
                    y_hat_[i] = runoff
                    y_t_[i] = y_t
                    # Updating the total mass of the system
                    self.mass_balance.change_mass(self.model)  # Updating global balance
                    time.sleep(0.01)
                self.mass_balance.report_mass(self.model)  # Global mass balance
            self.save_data(y_hat_)

        except KeyboardInterrupt:
            interrupt = True
            self.finalize(interrupt)
            log.info("You have entered CTRL+C.. Wait to finalize")

    def save_data(self, y_hat):
        y_hat_np = y_hat.detach().numpy()

        # Define the output directory
        dir_path = Path(self.cfg.synthetic.output_dir)
        # Check if the directory exists, if not, create it
        dir_path.mkdir(parents=True, exist_ok=True)

        # Define the output file path
        file_path = dir_path / self.cfg.synthetic.nams

        # Save the numpy array to the file
        np.save(file_path, y_hat_np)

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self) -> None:
        raise NotImplementedError

    def finalize(self, interrupt=False):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
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
