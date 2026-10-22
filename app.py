import gradio as gr
from fastapi_app.main import app, clean_encode_text, model, label_mapping, DEVICE
import torch

# Hàm xử lý riêng cho giao diện người dùng Gradio
def analyze_comment_ui(text):
    if not text.strip():
        return "Vui lòng nhập bình luận!"
    
    encoded = clean_encode_text(text)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_id = int(torch.argmax(output, dim=1).item())
        
    sentiment = label_mapping[pred_id]
    confidence = round(float(probs[pred_id]) * 100, 2)
    return f"Kết quả: {sentiment.upper()} (Độ tin cậy: {confidence}%)"

# Xây dựng giao diện Gradio
with gr.Blocks(title="YouTube Sentiment Insights") as demo:
    gr.Markdown("# YouTube Sentiment Analysis API & Demo")
    gr.Markdown(
        "Ứng dụng phân tích cảm xúc bình luận bằng Deep Learning Bi-LSTM.\n\n"
        "- **Swagger API Documentation:** [`/docs`](/docs)\n"
        "- **API Endpoint cho Extension:** `POST /predict`"
    )
    
    with gr.Row():
        input_text = gr.Textbox(label="Nhập bình luận để thử nghiệm", placeholder="Ví dụ: This video is really helpful!")
        output_text = gr.Label(label="Dự đoán")
        
    submit_btn = gr.Button("Phân tích cảm xúc")
    submit_btn.click(fn=analyze_comment_ui, inputs=input_text, outputs=output_text)

# Gắn FastAPI backend vào Gradio
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)