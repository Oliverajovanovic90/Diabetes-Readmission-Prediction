# ==============================
# Dockerfile for Diabetes Readmission Prediction API
# ==============================

# Use lightweight Python image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 9696

# Run FastAPI app
CMD ["python", "serve.py"]