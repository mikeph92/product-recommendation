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

# Install build dependencies
pip install --upgrade pip
pip install wheel setuptools
pip install maturin

# Install dependencies
pip install -r requirements.txt

# Clear any existing cache
rm -rf ~/.streamlit/cache/*

echo "Setup completed successfully!"
