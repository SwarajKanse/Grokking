#!/usr/bin/env bash
set -e

echo "=== Step 1: System packages ==="
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git htop tmux curl unzip build-essential

echo "=== Step 2: Install Azure CLI ==="
if ! command -v az &> /dev/null; then
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

echo "=== Step 3: Clone or update repo ==="
if [ ! -d "$HOME/Grokking" ]; then
    git clone https://github.com/SwarajKanse/Grokking.git "$HOME/Grokking"
else
    cd "$HOME/Grokking" && git pull origin main
fi

echo "=== Step 4: Python virtual environment ==="
if [ ! -d "$HOME/venv" ]; then
    python3 -m venv "$HOME/venv"
fi
source "$HOME/venv/bin/activate"
pip install --upgrade pip
pip install -r "$HOME/Grokking/requirements.txt" pyarrow

echo "=== Step 5: Verify Python ML Stack ==="
python3 -c "import pandas, numpy, sklearn, lightgbm, xgboost, catboost, rapidfuzz, jellyfish, pyarrow, tqdm; print('ALL LIBRARIES IMPORTED CLEANLY!')"

echo "=== VM SETUP FULLY COMPLETE ==="
