#!/usr/bin/env bash
set -euo pipefail

echo "Brain Tumor MRI Classifier"
echo "--------------------------"

if [ ! -f "deploy_effb3.pth" ]; then
  echo "ERROR: deploy_effb3.pth not found in project root."
  exit 1
fi

python3 -c "import fastapi, uvicorn, torch, timm, albumentations, PIL, numpy" >/dev/null 2>&1 || {
  echo "Installing Python dependencies..."
  python3 -m pip install -r requirements.txt
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
python3 inference.py

#!/bin/bash
# Quick Start Script for Brain Tumor Detection UI

echo "🧠 Brain Tumor Detection - Quick Start"
echo "======================================"
echo ""

# Check if model exists
if [ ! -f "deploy_effb3.pth" ]; then
    echo "❌ Error: deploy_effb3.pth not found!"
    echo "Please ensure the model file is in the project root."
    exit 1
fi

# Check if static folder exists
if [ ! -d "static" ]; then
    echo "📁 Creating static folder..."
    mkdir static
fi

echo "✅ Model file found: deploy_effb3.pth"
echo "✅ Static folder ready"
echo ""
echo "🚀 Starting FastAPI server..."
echo "   → Access the app at: http://localhost:8000"
echo "   → API docs at:      http://localhost:8000/docs"
echo "   → Press Ctrl+C to stop"
echo ""

# Ensure dependencies exist (best-effort)
python3 -c "import fastapi, uvicorn, torch, timm, albumentations, PIL, numpy" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
    echo ""
fi

# Run the server
python3 inference.py
