#!/bin/bash

# Create the virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install libraries
pip install -r requirements.txt

# Run pipeline
python src/main.py