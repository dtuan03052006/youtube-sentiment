FROM python:3.10-slim

# Ngăn sinh file .pyc thừa và xuất log trực tiếp ra console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Cài đặt các gói phụ thuộc
COPY requirements.txt .

# Cài PyTorch CPU siêu nhẹ (~180MB thay vì 2.5GB bản CUDA) để tiết kiệm RAM & Disk
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 2. Copy toàn bộ mã nguồn và trọng số mô hình
COPY fastapi_app/ /app/fastapi_app/

# Port chạy ứng dụng
EXPOSE 8000

CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]