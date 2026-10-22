FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Cài đặt PyTorch CPU với --extra-index-url (để vẫn tải được các gói phụ trợ từ PyPI)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện trong requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo user thông thường (UID 1000) theo đúng tiêu chuẩn Hugging Face Spaces
RUN useradd -m -u 1000 user && chown -R user:user /app

# Copy mã nguồn ứng dụng và model
COPY --chown=user:user fastapi_app/ /app/fastapi_app/

# Chuyển quyền sang user thường trước khi chạy app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces mở cổng 7860
EXPOSE 7860

CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "7860"]
