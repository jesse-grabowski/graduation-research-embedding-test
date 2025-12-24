FROM python:3.11-slim

WORKDIR /app

# CPU torch
RUN pip install --no-cache-dir torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
