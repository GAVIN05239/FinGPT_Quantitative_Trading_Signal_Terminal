FROM python:3.10-slim

# Set working directory
WORKDIR /app

# install the CPU version of PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies and install other packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and database
COPY . .

# Startup command
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
