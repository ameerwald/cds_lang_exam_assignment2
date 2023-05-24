#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code 
python3 src/logistic_regression_classifier.py
python3 src/neural_network_classifier.py

# deactive the venv
deactivate