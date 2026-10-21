FROM python:3.10-slim

# Ngăn sinh file .pyc và in log trực tiếp ra console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tạo user không phải root (UID 1000) theo chuẩn của Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy requirements và cài đặt
COPY --chown=user:user requirements.txt .

# 1. Cài PyTorch CPU siêu nhẹ (~180MB thay vì ~2.5GB bản CUDA)
# 2. Cài các gói còn lại
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn ứng dụng vào container
COPY --chown=user:user fastapi_app/ /app/fastapi_app/

# Port 7860 là cổng mặc định của Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "7860"]

CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "7860"]