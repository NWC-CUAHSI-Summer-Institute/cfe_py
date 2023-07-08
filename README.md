# dCFE
dCFE is a differentiable version of CFE. 
Developed as a folk of https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py
Both dCFE and cfe_py is for prototyping, research, and development.
The official CFE code lives here: https://github.com/NOAA-OWP/cfe/

## Installation

Use the package manager conda to create envrionment ```dCFE```

```bash
conda env create -f environment.yml
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)