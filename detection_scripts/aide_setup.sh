conda remove --name aide --all -y
conda create --prefix ./aide python=3.10 -y
conda activate ./aide
pip install -r requirements.txt
# export PIP_CACHE_DIR=<path>
pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
