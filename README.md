# FinalProject_CSCI7000

## Main project: Qwen 3.5 local planner

### Setup

```bash
conda create -n CSCI7000 python=3.11.11
conda activate CSCI7000

pip install uv
uv pip install -r requirements.txt


python main.py
python run_kitchen_agentic_benchmark.py
python smolagents_kitchen_demo.py

<!-- FOR LEROBOT AND LIBERO -->
Ubuntu:
wsl --install -d Ubuntu-24.04
wsl.exe -d Ubuntu-24.04
sudo apt update

sudo apt install -y git wget curl build-essential cmake pkg-config \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglfw3-dev \
  libosmesa6-dev libx11-dev libxext-dev libxi-dev libxrandr-dev \
  libxinerama-dev libxcursor-dev libxfixes-dev libglew-dev patchelf ffmpeg

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
conda create -n LEROBOT python=3.12



cd ~
rm -rf lerobot
git clone https://github.com/huggingface/lerobot.git
cd ~/lerobot


conda activate LEROBOT
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -e ".[libero]"

If you have problem with Cmake
export CMAKE_POLICY_VERSION_MINIMUM=3.5
python -m pip install --no-cache-dir egl_probe
python -m pip install --no-cache-dir -e ".[libero]"
echo 'export CMAKE_POLICY_VERSION_MINIMUM=3.5' >> ~/.bashrc



Accept Terms and Conditions here:
https://huggingface.co/google/paligemma-3b-pt-224
hf auth login

Create access token: Enable: Read access to contents of all public gated repos you can access

ls -l /dev/dri
sudo usermod -aG render,video $USER

exit or cntrl D:
wsl --shutdown

Boot Ubuntu back up again
Activate Venv through conda
run script: 
export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --output_dir="/mnt/d/Yuri/CU BOULDER/Masters/Sem 2/CSCI 7000/FinalProject_CSCI7000/Libero_Results/libero_smoke"
```













Running scripts:
python .\benchmark_vlm_models.py --annotations .\data\vlm_annotations.csv --use-memory --models qwen2.5-vl-3b-instruct
