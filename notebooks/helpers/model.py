
import torch.nn as nn
import torch
class CrossAttentionTextEnrichmentModel(nn.Module):
    def __init__(self, bert_embedding_dim, audio_embedding_dim, lstm_hidden_dim, num_classes,gru_hidden_dim,num_attention_layers=3):
        super(CrossAttentionTextEnrichmentModel, self).__init__()
        self.bert_embedding_dim = bert_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_classes = num_classes
        self.num_attention_layers = num_attention_layers
        # Cross-attention layer
        # self.cross_attention = nn.MultiheadAttention(embed_dim=bert_embedding_dim, num_heads=8)
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=bert_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(bert_embedding_dim) for _ in range(num_attention_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.5) for _ in range(num_attention_layers)])

        # Cross-attention layers for audio enriched with text
        self.audio_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=audio_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        self.audio_norm_layers = nn.ModuleList([nn.LayerNorm(audio_embedding_dim) for _ in range(num_attention_layers)])
        self.audio_dropout_layers = nn.ModuleList([nn.Dropout(0.5) for _ in range(num_attention_layers)])

        # GRU layer
        # self.gru = nn.GRU(input_size=bert_embedding_dim, hidden_size=gru_hidden_dim, 
        #                   batch_first=True, bidirectional=False, num_layers=1, dropout=0.5)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=bert_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=False, num_layers=1,dropout=0.5)
        self.audio_lstm = nn.LSTM(input_size=audio_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=False, num_layers=1, dropout=0.5)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim*2)
        # self.fc1 = nn.Linear(lstm_hidden_dim * 2, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, num_classes)
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_dim*2, num_classes)  # Bidirectional doubles hidden dim

        # Softmax for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, bert_embeddings, audio_embeddings,text_mask=None, audio_mask=None):
        """
        Args:
        - bert_embeddings: (batch_size, seq_len_text, bert_embedding_dim)
        - audio_embeddings: (batch_size, seq_len_audio, audio_embedding_dim)

        Returns:
        - logits: (batch_size, num_classes)
        """
        # Transpose embeddings to match MultiheadAttention input (seq_len, batch_size, embed_dim)
        bert_embeddings = bert_embeddings.permute(1, 0, 2)  # (seq_len_text, batch_size, bert_embedding_dim)
        audio_embeddings = audio_embeddings.permute(1, 0, 2)  # (seq_len_audio, batch_size, audio_embedding_dim)

        # Apply cross-attention
        # enriched_text_embeddings, _ = self.cross_attention(
        #     query=bert_embeddings,  # Texto como consulta
        #     key=audio_embeddings,   # Audio como clave
        #     value=audio_embeddings  # Audio como valor
        #     # key_padding_mask=audio_mask,  # Máscara para padding del audio
        # )
        for i in range(self.num_attention_layers):
            attention_output, _ = self.cross_attention_layers[i](
                query=bert_embeddings, 
                key=audio_embeddings, 
                value=audio_embeddings
                # key_padding_mask=audio_mask
            )
            # Apply normalization and dropout
            bert_embeddings = bert_embeddings + self.dropout_layers[i](attention_output)  # Residual Connection

            bert_embeddings = self.norm_layers[i](bert_embeddings)
        # Volver a (batch_size, seq_len_text, bert_embedding_dim)
        # enriched_text_embeddings = enriched_text_embeddings.permute(1, 0, 2)
        enriched_text_embeddings = bert_embeddings.permute(1, 0, 2)
          # Pass through LSTM
        lstm_out, _ = self.lstm(enriched_text_embeddings)  # (batch_size, seq_len_text, lstm_hidden_dim*2)

        
        # Take the final hidden state for classification
        lstm_out_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim*2)
        # Cross-attention for audio enriched with text
        for i in range(self.num_attention_layers):
            attention_output, _ = self.audio_attention_layers[i](
                query=audio_embeddings, 
                key=bert_embeddings, 
                value=bert_embeddings
            )
            audio_embeddings = audio_embeddings + self.audio_dropout_layers[i](attention_output)  # Residual Connection
            audio_embeddings = self.audio_norm_layers[i](audio_embeddings)

        enriched_audio_embeddings = audio_embeddings.permute(1, 0, 2)  # (batch_size, seq_len_audio, audio_embedding_dim)

        # Pass through audio LSTM
        audio_lstm_out, _ = self.audio_lstm(enriched_audio_embeddings)
        audio_lstm_out_last = audio_lstm_out[:, -1, :]  # Take the last hidden state
        combined_out = torch.cat((lstm_out_last, audio_lstm_out_last), dim=1)  # (batch_size, lstm_hidden_dim * 2)

        # Apply dropout
        combined_out = self.dropout(combined_out)
        # gru_out, _ = self.gru(enriched_text_embeddings) 
      

        # Apply layer normalization
        lstm_out_last = self.layer_norm(combined_out)
        # x = self.fc1(lstm_out_last)
        # x = nn.ReLU()(x)
        # x = self.fc2(x)
        # x = nn.ReLU()(x)
        # x = self.fc3(x)
        # x = nn.ReLU()(x)
        
        # Fully connected layer
        logits = self.fc(lstm_out_last)  # (batch_size, num_classes)

        # Apply softmax
        probabilities = self.softmax(logits)

        return probabilities