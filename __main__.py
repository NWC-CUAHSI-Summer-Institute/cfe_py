import hydra
import logging
from omegaconf import DictConfig
import time

from src.agents.DifferentiableCFE import DifferentiableCFE
from src.agents.SyntheticAgent import SyntheticAgent

log = logging.getLogger(__name__)

import os

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    print(f"Running in {cfg.run_type} mode")
    if (cfg.run_type == "ML") | (cfg.run_type == "ML_synthetic_test"):
        agent = DifferentiableCFE(cfg)  # For Running against Observed Data
    elif cfg.run_type == "generate_synthetic":
        agent = SyntheticAgent(cfg)
    agent.run()
    agent.finalize()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
