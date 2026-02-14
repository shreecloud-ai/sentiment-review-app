#!/bin/bash
set -e

# Start FastAPI in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for FastAPI to start
sleep 5

# Then start Streamlit (foreground)
exec streamlit run streamlit_app/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0