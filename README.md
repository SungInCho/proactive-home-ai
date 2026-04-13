## Setup

### 1. Clone the repository
git clone https://github.com/your-username/proactive-home-ai.git
cd proactive-home-ai
git submodule update --init --recursive

### 2. Download model weights
The weights file is not included in this repository due to file size limits.
Download manually and place it in the `weights/` folder:

mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

### 3. Install dependencies
pip install -r requirements.txt