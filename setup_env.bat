@echo off
echo === Creating virtual environment ===
python -m venv e2d_env

echo === Activating environment ===
call e2d_env\Scripts\activate.bat

echo === Upgrading pip ===
python -m pip install --upgrade pip

echo === Installing PyTorch with CUDA 12.4 (compatible with your CUDA 12.7 driver) ===
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

echo === Installing remaining dependencies ===
pip install avalanche-lib==0.5.0 Pillow numpy tqdm

echo === Verifying GPU is available ===
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ===================================
echo Environment setup complete!
echo To activate: e2d_env\Scripts\activate.bat
echo ===================================
pause
