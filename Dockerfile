FROM python:3.10-slim

# 1. Install ALL required system dependencies for OpenCV and Media processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory to the project root
WORKDIR /usr/src/app

# 3. Copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application
COPY . .

# 5. Set Environment variables for Railway
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 6. Start the app using a shell to correctly map the Railway $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]