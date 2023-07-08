# dCFE
dCFE is a differentiable version of CFE, which allows gradient tracking from runoff to 9  calibration parameters.  

Both dCFE and cfe_py is for prototyping, research, and development. This code is folked and developed from https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py. The official CFE code lives here: https://github.com/NOAA-OWP/cfe/

## Installation

Use the package manager conda to create envrionment ```dCFE```

```bash
conda env create -f environment.yml
```

## Usage
- Change ```config.yaml```, ```src/models/config/```, ```src/data/config/``` to change configuration
- run __main__.py function

```python
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
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)