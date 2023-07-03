import hydra
import logging
from omegaconf import DictConfig
import time

from src.agents.DifferentiableCFE import DifferentiableCFE

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    agent = DifferentiableCFE(cfg)  # For Running against Observed Data
    agent.run()
    agent.finalize()
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()