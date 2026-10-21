import torch
import torch.nn as nn

class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, num_layers=2, dropout=0.3):
        super(SentimentBiLSTM, self).__init__()
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 2. Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]
        
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Lấy hidden state cuối cùng của cả 2 chiều: forward (tầng cuối) và backward (tầng cuối)
        # hidden[-2, :, :] là chiều forward của layer cuối, hidden[-1, :, :] là backward
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, hidden_dim * 2]
        
        out = self.fc(self.dropout(hidden_last))  # [batch_size, output_dim]
        return out

# Alias để tương thích tên gọi
SentimentBiLSTMWithAttention = SentimentBiLSTM
