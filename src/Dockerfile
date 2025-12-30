# Base image nhẹ, đủ xài cho pandas
FROM python:3.10-slim

# Set working directory trong container
WORKDIR /app

# Cài system dependencies (parquet cần)
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*


# Cài trực tiếp lib cần dùng
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    fastparquet \
    matplotlib \
    seaborn \
    scipy \
    psutil

# Copy source code
COPY src/ /app/src/
COPY data/ /app/data/
COPY images/ /app/images
# # Set PYTHONPATH để import được src.handle.fill_null
# ENV PYTHONPATH=/app

# # Command chạy chính
# CMD ["python", "src/main.py"]
 # cho container ngủ
CMD ["tail", "-f", "/dev/null"]