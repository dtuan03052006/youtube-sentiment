import os
import re
import pickle
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Import kiến trúc mô hình từ file model_def.py vừa tạo ở Bước 2
from fastapi_app.model_def import SentimentBiLSTMWithAttention

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API for analyzing the sentiment of YouTube comments",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình phục vụ giao diện trang web
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
async def serve_homepage():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "API online. Go to /docs for Swagger documentation."}


class CommentRequest(BaseModel):
    comments: List[str]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
vob_path = os.path.join(os.path.dirname(__file__), "vocab(1).pkl")

with open(vob_path, "rb") as f:
    vocab = pickle.load(f)

model = SentimentBiLSTMWithAttention(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    output_dim=3,
    num_layers=2,
    dropout=0.3
)
model = model.to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
def clean_encode_text(text,max_len=100):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens= text.split()
    token_ids =[]
    for token in tokens:
        token_ids.append(vocab.get(token,vocab["<UNK>"]))
    if max_len > len(token_ids):
        padding = max_len-len(token_ids)
        token_ids.extend([vocab["<PAD>"]]* padding)
    else:
        token_ids= token_ids[:max_len]
    return token_ids

@app.post("/predict")
async def predict_sentiment(request: CommentRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided.")
    encode_batch = [clean_encode_text(c) for c in request.comments]
    input_tensor = torch.tensor(encode_batch, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probability = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=1).cpu().numpy()
    results = []
    for comment, pred_id,pros in zip(request.comments, predictions, probability.cpu().numpy()):
        results.append({
            "comment": comment,
            "label_id" : int(pred_id),
            "label": label_mapping[int(pred_id)],
            "confidence": round(float(pros[pred_id]), 4)
        })
    return results