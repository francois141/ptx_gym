sudo apt-get update && sudo apt-get install -y build-essential python3-dev git
sudo apt-get install -y ccache
python3 -m venv venv_triton
source venv_triton/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
git clone https://github.com/openai/triton.git
cd triton
git fetch --tags 
git checkout v3.6.0
pip install -v -e .

python3 -c "import triton; print(triton.__version__)"