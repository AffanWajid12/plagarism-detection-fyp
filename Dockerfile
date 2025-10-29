# ----------------------------------------------------------
# Base image: PyTorch with CUDA 12.1 (GPU-enabled, includes Python 3.12)
# ----------------------------------------------------------
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ----------------------------------------------------------
# Set working directory
# ----------------------------------------------------------
WORKDIR /app

# ----------------------------------------------------------
# Install system dependencies
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Copy project files
# ----------------------------------------------------------
COPY . .

# ----------------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# Expose Flask port
# ----------------------------------------------------------
EXPOSE 5000

# ----------------------------------------------------------
# Environment variables
# ----------------------------------------------------------
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models_cache

# ----------------------------------------------------------
# Create uploads folder
# ----------------------------------------------------------
RUN mkdir -p uploads

# ----------------------------------------------------------
# Run the Flask app
# ----------------------------------------------------------
CMD ["python", "app.py"]
