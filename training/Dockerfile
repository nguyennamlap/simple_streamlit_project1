# 1. Base image
FROM python:3.11-slim

# 2. Set workdir
WORKDIR /app

# 3. Copy requirements (ta sẽ tạo file requirements.txt)
COPY training/requirements.txt /app

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
COPY training /app/training
COPY data/ /app/data/
# 6. Set environment variables (tuỳ chọn)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 7. Command to run
CMD tail -l /dev/null
