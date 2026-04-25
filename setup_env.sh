sudo apt-get update && sudo apt-get install -y build-essential python3-dev git
sudo apt-get install -y ccache
python3 -m venv venv_triton
source venv_triton/bin/activate
pip install --upgrade pip setuptools wheel
pip install pandas matplotlib
pip install torch torchvision torchaudio
git clone https://github.com/francois141/triton_ptx/
cd triton_ptx
pip install -v -e .

python3 -c "import triton; print(triton.__version__)"
git config --global user.name "Francois Costa"
git config --global user.email "frankost.costa@gmail.com"
