#!/usr/bin/env bash
set -e

ENV_DIR=".venv"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete."