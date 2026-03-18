FROM python:3.10-slim

# Install system dependencies for OpenCV and HDF5
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY . .

# Set Environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
# Force TensorFlow to use CPU to avoid CUDA errors in logs
ENV TF_CPP_MIN_LOG_LEVEL=2 

EXPOSE 8000

# Start the app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]