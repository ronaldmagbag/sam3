# Docker Image with CUDA 12.6

This document describes how to build and use the Docker image with CUDA 12.6 and Ubuntu 22.04.

## Requirements

This Docker image is configured to meet SAM3's requirements:
- **Python**: 3.12 or higher
- **PyTorch**: 2.7 or higher
- **CUDA**: 12.6 or higher

## Prerequisites

- Docker installed with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.6 support
- Docker with GPU support enabled

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Building the Image

### Using PowerShell (Windows)

```powershell
.\build_docker_cuda126.ps1
```

### Using Bash (Linux/Mac)

```bash
chmod +x build_docker_cuda126.sh
./build_docker_cuda126.sh
```

### Manual Build

```bash
docker build -f Dockerfile.cuda126 -t geoseg-cuda126:latest .
```

## Running the Container

### Basic Run

```bash
docker run --gpus all -it --rm geoseg-cuda126:latest
```

### With Volume Mounts

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -p 8000:8000 \
  -p 8888:8888 \
  geoseg-cuda126:latest
```

### PowerShell (Windows)

```powershell
docker run --gpus all -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -p 8000:8000 `
  -p 8888:8888 `
  geoseg-cuda126:latest
```

## Running Tests

### Test SAM3 Image Processing

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  geoseg-cuda126:latest \
  python tests/test_image_sam3.py
```

### Test SAM3 Video Processing

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  geoseg-cuda126:latest \
  python tests/test_video_sam3.py
```

## Running Training

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  geoseg-cuda126:latest \
  python src/training/train_model.py --data ./data/dataset_versions/v1 --save ./models
```

## Running API Server

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  geoseg-cuda126:latest \
  python src/deployment/run_endpoint.py --model ./models --host 0.0.0.0 --port 8000
```

## Jupyter Notebook

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -p 8888:8888 \
  geoseg-cuda126:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Verify CUDA Installation

```bash
docker run --gpus all --rm geoseg-cuda126:latest nvidia-smi
docker run --gpus all --rm geoseg-cuda126:latest python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## Image Specifications

- **Base Image**: `nvidia/cuda:12.6.0-devel-ubuntu22.04`
- **Python**: 3.12 (required by SAM3)
- **PyTorch**: >=2.7.0 with CUDA 12.6 support (required by SAM3)
- **CUDA**: 12.6 (required by SAM3)
- **SAM3**: Installed in editable mode from `3rdparty/sam3`

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed and Docker is restarted:
```bash
sudo systemctl restart docker
```

### CUDA Version Mismatch

Verify your host CUDA version matches the container:
```bash
nvidia-smi
```

### Out of Memory

Reduce batch size or use a smaller model variant.

### Permission Issues

On Linux, you may need to add your user to the docker group:
```bash
sudo usermod -aG docker $USER
```

## Notes

- The image includes all dependencies for SAM3, training, and deployment
- Data, models, and logs should be mounted as volumes to persist across container runs
- Port 8000 is for the API server, port 8888 is for Jupyter notebooks

