FROM python:3.10-slim

# Không tạo pycache, log gọn
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (plotly + sklearn cần)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY streamlit/requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY streamlit/ /app/streamlit/
COPY data/ /app/data/

# Expose Streamlit port
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "/app/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

