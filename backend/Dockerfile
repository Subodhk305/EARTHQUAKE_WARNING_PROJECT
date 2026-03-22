FROM python:3.11-slim

# System dependencies for ObsPy + PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev libgeos-dev libproj-dev \
    gfortran libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create model directory
RUN mkdir -p earthquake_service/models/saved

EXPOSE 8000

CMD ["uvicorn", "earthquake_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
