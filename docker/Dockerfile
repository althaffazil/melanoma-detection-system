# -------- Base Image --------
FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]