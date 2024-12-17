from torch import nn
import torch
import torch.nn.functional as F
class EnhancedCLAPModel(nn.Module):
    def __init__(self, input_dim=768, projection_dim=256, lstm_hidden_dim=256, num_heads=8):
        super(EnhancedCLAPModel, self).__init__()
        
        # Proyecciones iniciales
        self.audio_projection = nn.Linear(input_dim, projection_dim)
        self.text_projection = nn.Linear(input_dim, projection_dim)
        
        # Cross Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads)
        
        # LSTM para procesar secuencias temporales de audio
        self.audio_lstm = nn.LSTM(input_size=projection_dim, hidden_size=lstm_hidden_dim, 
                                  batch_first=True, bidirectional=True)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, projection_dim),  # x2 por bidireccional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Regularización
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, audio_embeddings, text_embeddings):
        # Proyecciones iniciales
        audio_proj = self.audio_projection(audio_embeddings)  # (batch, seq_len_audio, proj_dim)
        text_proj = self.text_projection(text_embeddings)    # (batch, proj_dim)
        
        # Atención cruzada: texto guía al audio
        audio_proj, _ = self.cross_attention(audio_proj, text_proj.unsqueeze(0), text_proj.unsqueeze(0))
        
        # Procesamiento temporal con LSTM (solo para audio)
        audio_proj, _ = self.audio_lstm(audio_proj)
        audio_proj = audio_proj[:, -1, :]  # Tomar el último estado oculto
        
        # Refinar embeddings con FFN
        audio_proj = self.ffn(self.dropout(audio_proj))
        text_proj = self.ffn(self.dropout(text_proj))
        
        # Normalización para similitud de coseno
        audio_proj = F.normalize(audio_proj, dim=1)
        text_proj = F.normalize(text_proj, dim=1)
        
        # Similitud de coseno
        similarity_matrix = torch.matmul(audio_proj, text_proj.T)
        
        return similarity_matrix