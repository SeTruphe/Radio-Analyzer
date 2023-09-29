#!/bin/bash
echo "Creating environment in .venv"
python3 -m venv .venv
echo "Activating environment"
source .venv/bin/activate
echo "Installing Radio Analyzer"
pip install .