#!/bin/bash
# Downloads the B200 LEGO detection dataset from Kaggle into data/raw (about 14 GB).
# The dataset is public so you don't need a Kaggle API key.
# Run from the repo folder with the venv activated:  bash scripts/download_data.sh

mkdir -p data/raw
kaggle datasets download ronanpickell/b100-lego-detection-dataset -p data/raw --unzip
