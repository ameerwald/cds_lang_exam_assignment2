#!/usr/bin/env bash

# create virtual environment
python3 -m venv env

#activate virtual environment
source ./env/bin/activate

# then install requirements
python3 -m pip install --upgrade pip
pip install -U numpy scipy scikit-learn
python3 -m pip install -r requirements.txt

deactivate
