FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Models directory (will be mounted as a volume in production)
RUN mkdir -p models

EXPOSE 8001

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8001"]
