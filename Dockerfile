FROM python:3.10-slim

# Ngăn Python tạo file cache .pyc và cho phép log in ra ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Hugging Face yêu cầu chạy dưới user thông thường (UID 1000) vì lý do bảo mật
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Cài đặt PyTorch CPU siêu nhẹ (~150MB thay vì >2GB bản CUDA) để tiết kiệm tài nguyên
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện cần thiết trong requirements.txt
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code và model weights
COPY --chown=user:user fastapi_app/ /app/fastapi_app/

# Hugging Face Space mặc định mở cổng 7860
EXPOSE 7860

# Lệnh khởi chạy server FastAPI với Uvicorn trên cổng 7860
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "7860"]

