FROM python:3.10-slim

# Install system dependencies (OpenCV fix)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# FIX: use PORT safely
ENV PORT=8501

CMD ["bash", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
