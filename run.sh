#!/usr/bin/env bash
set -euo pipefail

echo "Brain MRI AI Suite (Detection + Segmentation)"
echo "---------------------------------------------"

if [ ! -f "deploy_effb3.pth" ]; then
  echo "ERROR: deploy_effb3.pth not found in project root."
  exit 1
fi

if [ ! -f "models/segmentation/best_unet_model.pth" ]; then
  echo "ERROR: models/segmentation/best_unet_model.pth not found."
  echo "Place the segmentation checkpoint at: models/segmentation/best_unet_model.pth"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv (.venv)..."
  python3 -m venv .venv
fi

source .venv/bin/activate

python -c "import fastapi, uvicorn, torch, timm, albumentations, PIL, numpy, cv2, segmentation_models_pytorch" >/dev/null 2>&1 || {
  echo "Installing Python dependencies into .venv ..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
}

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

echo "Building frontend into static/ ..."
rm -rf static
mkdir -p static
(cd frontend && npm run build)
cp -R frontend/dist/* static/

echo "Starting API server on http://localhost:8000"
python inference.py
