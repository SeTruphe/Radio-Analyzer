#!/bin/bash

VENV_PATH=".venv"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating Environment"
    source $VENV_PATH/bin/activate
else
    echo "Virtual Environment already active"
fi
echo "Running Radio Analyzer"
radio-analyzer