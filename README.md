# dCFE

The differentiable parameter learning Conceptual Functional Equivalent (dCFE) is a differentiable implementation of CFE (see below). All operations of the model are coded using PyTorch to track gradients and tune model parameters. Currently the model allows gradient tracking from runoff to 9 calibration parameters (```bb```, ```satdk```, ```smcmax```, ```slop```, ```Cgw```, ```expon```, ```max_gw_storage```, ```K_nash```, ```K_lf```).  

#### Conceptual Functional Equivalent (CFE) Model
The CFE model is designed to be a simplified and functionaly equivalent model of the National Water Model. The model code was originally written by Dr. Fred Ogden and converted to BMI-compliant format in the Next-Gen framework by NOAA-OWP. The official CFE code by Dr. Fred Oden and NOAA-OWP lives [here](https://github.com/NOAA-OWP/cfe/).  [The Python version of the code](https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py) is developed for the prototyping, research, and development. This code is developed upon the Python version and for research purpose. 

## Installation
Use conda to create your own env based on our ```environment.yml``` file

```bash
conda env create -f environment.yml
conda activate dCFE
```

## Running this code
We are using [Hydra](https://github.com/facebookresearch/hydra) to store/manage configuration files.

The main branch code is currently configured to run the CAMELS basin id #0102500 test case. If you want to use your own case, you will need to manage three config files located here:

- ```dCFE/config.yaml```
    - The main config file. I recommend looking at the Hydra config docs here to learn how this file is structured.
- ```dpLGAR/src/models/config/base.yaml```
    - This holds all config values for the models
- ```dpLGAR/src/data/config/<site_name>.yaml```
    - This holds all config values for the dataset you're working on.
- ```dpLGAR/data/```
    - This holds all forcing (P, PET) and validation dataset (runoff)

To run the code, just run the following command inside the dpLGAR/ folder:

```python  __main__.py```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)