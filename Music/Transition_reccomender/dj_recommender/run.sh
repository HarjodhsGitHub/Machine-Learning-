#!/bin/bash

# DJ Recommender Startup Script (macOS/Linux)

echo "🎵 Starting DJ Transition Recommender..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p backend uploads data

# Start backend
echo "🚀 Starting FastAPI backend on http://localhost:8000"
cd backend
python main.py
