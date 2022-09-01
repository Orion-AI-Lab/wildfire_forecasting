[![DOI](https://zenodo.org/badge/489363446.svg)](https://zenodo.org/badge/latestdoi/489363446)


# Intro 

This repository contains the code to reproduce the figures and experiments in our paper [Wildfire Danger Prediction and Understanding with Deep Learning](https://doi.org/10.1029/2022gl099368), published in [Geophysical Research Letters](https://agupubs.onlinelibrary.wiley.com/journal/19448007).

Authored by Spyros Kondylatos, Ioannis Prapas, Michele Ronco, Ioannis Papoutsis, Gustau Camps-Valls, Maria Piles, Miguel-Angel Fernandez-Torres, Nuno Carvalhais

# Setting up

## Installing the project


Now your project can be installed from local files:
```yaml
pip install -e .
```

Or directly from git repository:
```yaml
pip install git+https://github.com/Orion-AI-Lab/wildfire_forecasting --upgrade
```

So any file can be easily imported into any other file like so:
```python
from wildfire_forecasting.models.greece_fire_models import LSTM_fire_model
from wildfire_forecasting.datamodules.greecefire_datamodule import FireDSDataModule
```

## Installing the environment

The code has been tested in Python 3.8

`pip install -r requirements.txt`

## Downloading the data

Download the `datasets.tar.gz` from [https://zenodo.org/record/6528394](https://zenodo.org/record/6528394) and decompress it in your filesystem:

`tar -xzf datasets.tar.gz`

Add the path to the decompressed directory to  [configs/datamodule/fireds_spatiotemporal_datamodule.yaml](configs/datamodule/fireds_spatiotemporal_datamodule.yaml) and [configs/datamodule/fireds_temporal_datamodule.yaml](configs/datamodule/fireds_temporal_datamodule.yaml)

IMPORTANT NOTE: Make sure to have enough space to decompress the data. At least 250GB are needed!

# Running the code

The code is GPU-ready, and it is recommended to have a cuda-enabled NVIDIA GPU to run the experiments. 
They can also be run in a CPU, but expect slow training times

The code has been tested in a server with 128GB RAM and an NVIDIA RTX 3080 (10GB).

## Running the Random Forest model

See notebook [notebooks/RF.ipynb](notebooks/RF.ipynb).

## Training the LSTM model

Training the LSTM with the hyperparameters that were used in the paper:

`python run.py experiment=lstm_temporal_cls`

## Training the convLSTM model

Training the convLSTM with the hyperparameters that were used in the paper:

`python run.py experiment=clstm_spatiotemporal_cls`

# Custom Training

Please refer to the [README_template.md](README_template.md) of the code template to understand the code structure and perform any custom training.

# How to cite

Kondylatos, S., Prapas, I., Ronco, M., Papoutsis, I., Camps-Valls, G., Piles, M., et al. (2022). Wildfire Danger Prediction and Understanding with Deep Learning. Geophysical Research Letters, 49, e2022GL099368. https://doi.org/10.1029/2022GL099368 

# Acknowledgements

This repo uses the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). 

