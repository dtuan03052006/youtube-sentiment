import gradio as gr
from fastapi_app.main import app, clean_encode_text, model, label_mapping
import torch

# Hàm xử lý riêng cho giao diện người dùng Gradio
def analyze_comment_ui(text):
    if not text or not text.strip():
        return {}
    
    encoded = clean_encode_text(text)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
    # Trả về dict phân phối xác suất cho gr.Label (hiển thị thanh % trực quan)
    return {
        label_mapping[0].capitalize(): float(probs[0]),
        label_mapping[1].capitalize(): float(probs[1]),
        label_mapping[2].capitalize(): float(probs[2])
    }

# Xây dựng giao diện Gradio
with gr.Blocks(title="YouTube Sentiment Insights", theme=gr.themes.Soft(primary_hue="red", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎬 YouTube Sentiment Analysis API & Demo")
    gr.Markdown(
        "Ứng dụng phân tích cảm xúc bình luận bằng mô hình Deep Learning **Bi-LSTM**.\n\n"
        "- **Tài liệu Swagger API:** [`/docs`](/docs)\n"
        "- **API Endpoint cho Extension/Web:** `POST /predict`"
    )
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Nhập bình luận YouTube để thử nghiệm", 
            placeholder="Ví dụ: This video is really helpful and well explained!",
            lines=3
        )
        output_label = gr.Label(num_top_classes=3, label="Kết quả dự đoán")
        
    submit_btn = gr.Button("🚀 Phân tích cảm xúc", variant="primary")
    submit_btn.click(fn=analyze_comment_ui, inputs=input_text, outputs=output_label)

    gr.Examples(
        examples=[
            ["This video is really amazing, thank you so much!"],
            ["Worst content ever, completely useless and boring."],
            ["The video starts at 02:30 on Monday."]
        ],
        inputs=input_text,
        outputs=output_label,
        fn=analyze_comment_ui,
        cache_examples=False
    )

# Gắn FastAPI backend vào Gradio
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
