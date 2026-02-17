#!/bin/bash
# Run the Whiteboard Animator web app

cd "$(dirname "$0")"

# Install dependencies if needed
pip install -q -r requirements.txt

# Run the server
echo "🚀 Starting Whiteboard Animator on http://localhost:8080"
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
