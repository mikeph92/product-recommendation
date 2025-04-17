#!/bin/bash

# Set environment variables for memory management
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1  # Disable GPU to save memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Create necessary directories
mkdir -p .streamlit

# Configure memory limits for Python
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install maturin first
pip install maturin

# Install dependencies
pip install -r requirements.txt

# Install TensorFlow
pip install tensorflow==2.19.0

# Install TensorFlow Model Optimization
pip install tensorflow-model-optimization==0.7.5

# Install wheel and setuptools
pip install wheel==0.42.0
pip install setuptools==69.1.1

# Deactivate virtual environment
deactivate

# Clear any existing cache
rm -rf ~/.streamlit/cache/*

echo "Setup completed successfully!"
