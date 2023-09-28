#!/bin/bash

VENV_PATH=".venv"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual Environment not activated. Activating Environment"
    source $VENV_PATH/bin/activate
else
    echo "Virtual Environment already active"
fi
radio-analyzer