@echo off

:: Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

:: Install TensorFlow
pip install tensorflow==2.19.0

:: Install TensorFlow Model Optimization
pip install tensorflow-model-optimization==0.7.5

:: Install wheel and setuptools
pip install wheel==0.42.0
pip install setuptools==69.1.1

:: Deactivate virtual environment
deactivate 