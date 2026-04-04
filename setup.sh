rm -rf data
git clone https://huggingface.co/sproos/parameter-golf-tokenizers data
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt 
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git config --global user.name "Spruce Campbell" && git config --global user.email "spruce@mit.edu"
git clone https://github.com/dao-ailab/flash-attention
sudo apt update && sudo apt install git-lfs
cd flash-attention/hopper
MAX_JOBS=64 \
FLASH_ATTENTION_DISABLE_BACKWARD=FALSE \
FLASH_ATTENTION_DISABLE_SPLIT=FALSE \
FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
FLASH_ATTENTION_DISABLE_APPENDKV=FALSE \
FLASH_ATTENTION_DISABLE_LOCAL=FALSE \
FLASH_ATTENTION_DISABLE_SOFTCAP=FALSE \
FLASH_ATTENTION_DISABLE_PACKGQA=FALSE \
FLASH_ATTENTION_DISABLE_FP16=FALSE \
FLASH_ATTENTION_DISABLE_FP8=FALSE \
FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
FLASH_ATTENTION_DISABLE_CLUSTER=FALSE \
FLASH_ATTENTION_DISABLE_HDIM64=FALSE \
FLASH_ATTENTION_DISABLE_HDIM96=FALSE \
FLASH_ATTENTION_DISABLE_HDIM128=FALSE \
FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
FLASH_ATTENTION_DISABLE_SM80=FALSE \
python setup.py install
cd data
git lfs pull -I "datasets/fineweb10B_sp4096/*"
git lfs pull -I "tokenizers/fineweb_4096_bpe.model"