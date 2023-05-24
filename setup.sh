#!/usr/bin/env bash

# install venv
sudo apt-get update -y
sudo apt-get install python3-venv -y

# create virtual environment
python3 -m venv env

#activate virtual environment
source ./env/bin/activate

# then install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

deactivate