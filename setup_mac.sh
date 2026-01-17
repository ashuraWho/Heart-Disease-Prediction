#!/bin/bash

# ==============================================================================
# HEART DISEASE PREDICTION - MacOS Setup Script (V2)
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

# 2. Activation instructions
echo "[2/4] Activation Instructions:"
echo "    >>> To finish setup, you MUST RUN these commands manually in your terminal:"
echo "    >>> 1. conda activate heart_disease"
echo "    >>> 2. python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib shap tensorflow rich"

# 3. Special instructions for Apple Silicon (M1/M2/M3/M4)
echo "[3/4] Note for Apple Silicon Users:"
echo "    If you have an M1/M2/M3/M4 Mac, for better performance run:"
echo "    python -m pip install tensorflow-macos tensorflow-metal"

# 4. Critical Warning
echo "[4/4] CRITICAL EXECUTION WARNING:"
echo "    NEVER run the code with: /opt/anaconda3/bin/python ..."
echo "    ALWAYS run it with: python main.py"
echo "    (After activating the environment)"

echo "--------------------------------------------------------"
echo "To verify your environment later, run: which python"
echo "It should point to: .../envs/heart_disease/bin/python"
echo "--------------------------------------------------------"
