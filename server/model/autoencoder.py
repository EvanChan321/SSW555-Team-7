import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EEGTransformer(nn.Module):
    def __init__(self, num_classes, input_channels=64, sequence_length=795, d_model=64, num_heads=8, num_encoder_layers=3):
        super(EEGTransformer, self).__init__()
        self.embedding = nn.Linear(input_channels * sequence_length, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(64 * 795, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.position_encoding = PositionalEncoding(d_model, sequence_length)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dim_feedforward=128,  # You can adjust this feedforward dimension
            ),
            num_encoder_layers,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),  # Adjusted the input size here
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.position_encoding(x)
        
        x = x.permute(1, 0, 2)  # Swap batch and sequence length dimensions
        x = self.transformer_encoder(x)

        x = x.squeeze(dim=1)
        x = self.classifier(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # print(x.shape, self.pe[:, :x.size(1)].shape)
 
        x = x + self.pe[:, :x.size(1)]
        return x