#!/bin/bash

# ==============================================================================
# HEART DISEASE PREDICTION - MacOS Setup Script
# ==============================================================================
# This script helps resolve segmentation faults often caused by Anaconda's
# base environment conflicts on macOS.
# ==============================================================================

echo "--------------------------------------------------------"
echo "Starting MacOS/Anaconda Environment Setup"
echo "--------------------------------------------------------"

# 1. Create a fresh conda environment
echo "[1/4] Creating a clean conda environment 'heart_disease'..."
conda create -n heart_disease python=3.10 -y

# 2. Activate the environment (note: this might require user interaction outside the script)
echo "[2/4] Activation instructions:"
echo "    >>> To proceed, please run: conda activate heart_disease"
echo "    >>> Then run the following commands manually."

# 3. Install dependencies via PIP (to avoid MKL/Anaconda binary conflicts)
echo "[3/4] Installing dependencies via pip..."
# We use PIP instead of CONDA for libraries to avoid the dreaded MKL conflict on Mac
pip install pandas numpy matplotlib seaborn scikit-learn joblib shap tensorflow

# 4. Special instructions for Apple Silicon (M1/M2/M3/M4)
echo "[4/4] Note for Apple Silicon Users:"
echo "    If you have an M1/M2/M3/M4 Mac, for better performance run:"
echo "    pip install tensorflow-macos tensorflow-metal"

echo "--------------------------------------------------------"
echo "Setup script finished."
echo "CRITICAL: Remember to ALWAYS activate the environment before running the code:"
echo "          conda activate heart_disease"
echo "--------------------------------------------------------"
