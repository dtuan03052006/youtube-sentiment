import os
import re
import pickle
import torch
import gradio as gr
from fastapi.responses import FileResponse

# Import các thành phần từ fastapi_app
from fastapi_app.main import app as fastapi_app, clean_encode_text, model, label_mapping, device

# ==========================================================
# 1. Hàm phân tích cảm xúc phục vụ Gradio UI
# ==========================================================
def predict_sentiment(comment: str):
    if not comment or not comment.strip():
        return {"Vui lòng nhập bình luận": 1.0}
    
    tokens = clean_encode_text(comment)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Trả về phân phối xác suất cho 3 nhãn
    return {
        label_mapping[0].capitalize(): float(probabilities[0]),
        label_mapping[1].capitalize(): float(probabilities[1]),
        label_mapping[2].capitalize(): float(probabilities[2])
    }

# ==========================================================
# 2. Xây dựng giao diện Gradio chuyên nghiệp
# ==========================================================
custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 900px !important; margin: 0 auto !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", secondary_hue="slate"), css=custom_css, title="YouTube Sentiment AI") as demo:
    gr.Markdown(
        """
        # 🎬 YouTube Sentiment Analysis AI
        Phân tích cảm xúc bình luận YouTube thời gian thực sử dụng mạng nơ-ron **PyTorch BiLSTM**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            comment_input = gr.Textbox(
                lines=4,
                placeholder="Nhập hoặc dán bình luận YouTube vào đây...",
                label="Bình luận (Comment)"
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                submit_btn = gr.Button("🚀 Phân tích cảm xúc", variant="primary")
        with gr.Column(scale=2):
            output_label = gr.Label(num_top_classes=3, label="Phân phối cảm xúc dự đoán")

    submit_btn.click(fn=predict_sentiment, inputs=comment_input, outputs=output_label)
    clear_btn.click(lambda: "", outputs=comment_input)

    gr.Examples(
        examples=[
            ["This video is really amazing, thank you so much for sharing!"],
            ["This is the worst content I have ever seen, completely useless."],
            ["The video was released on Monday at 3pm."]
        ],
        inputs=comment_input,
        outputs=output_label,
        fn=predict_sentiment,
        cache_examples=False
    )

    gr.Markdown(
        """
        ---
        💡 *Hệ thống đồng thời phục vụ API tại `/predict` và Dashboard tùy chỉnh tại `/web`.*
        """
    )

# ==========================================================
# 3. Mount thêm Custom Web Dashboard và API FastAPI
# ==========================================================
static_dir = os.path.join(os.path.dirname(__file__), "fastapi_app", "static")
@fastapi_app.get("/web")
def serve_web_dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Custom UI not found"}

# Kết hợp Gradio và FastAPI vào biến `app` chuẩn của Hugging Face
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch()

