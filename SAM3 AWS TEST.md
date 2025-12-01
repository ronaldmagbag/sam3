SAM3 AWS TEST

ssh -i eks.pem ubuntu@ec2-18-207-218-133.compute-1.amazonaws.com

git clone --recursive https://github.com/ronaldmagbag/sam3.git
cd sam3

# 24.04
sudo apt-get update && sudo apt install -y python3-pip

sudo apt install -y python3.12-venv
python3.12 -m venv sam3env
source sam3env/bin/activate

pip install --upgrade pip
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install -e ".[train,dev]"
pip install einops decord psutil

pip install --upgrade huggingface_hub
hf auth login

python sam3/train/train.py -c configs/roboflow_v100/aws_test.yaml --use-cluster 0 --num-gpus 1
python tests/test_image_sam3.py tests/images/test.png "a building"


# Move pre-saved models to nvme
sudo mkdir -p /opt/dlami/nvme/sam3_checkpoints
mv checkpoint.pt /opt/dlami/nvme/sam3_checkpoints/
ln -s /opt/dlami/nvme/sam3_checkpoints/checkpoint.pt ~/sam3/datasets/logs/3d/checkpoints/checkpoint.pt



# 22.04
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev curl

sudo apt install -y python3.12 python3.12-venv python3.12-distutils python3.12-dev
