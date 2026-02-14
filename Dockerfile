# Use official Python slim image (smaller than full python)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first → better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn streamlit

# Copy the actual application code
COPY app/ ./app/
COPY streamlit_app/ ./streamlit_app/
COPY models/ ./models/
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose both ports (Render will use $PORT for one, but we expose both)
EXPOSE 8000 8501

# Use our custom entrypoint
ENTRYPOINT ["./entrypoint.sh"]