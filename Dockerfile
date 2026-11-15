# Start with a lightweight Python image
# This gives our container a Python runtime
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the dependency list before copying application code
COPY requirements.txt .

# avoid retaining pip's download cache in the image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, training script, and startup script into /app
COPY main.py .
COPY train.py .
COPY start.sh .

# Make the startup script executable
RUN chmod +x start.sh

# Document the port used by FastAPI application
EXPOSE 8000

# Start MLFlow server, train model, then launch FastAPI
CMD ["./start.sh"]
