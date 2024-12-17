import torch.nn as nn
import torch
class ImprovedAttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(ImprovedAttentionPooling, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),  # Reduce dimensionalidad
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)  # Genera el peso de atención
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        x: Tensor de forma (batch_size, seq_len, embedding_dim)
        """
        # Calcular pesos de atención
        attention_scores = torch.softmax(self.attention_weights(x), dim=1)  # (batch_size, seq_len, 1)

        # Generar representación ponderada
        context_vector = torch.sum(attention_scores * x, dim=1)  # (batch_size, embedding_dim)

        # Normalización final
        return self.norm(context_vector)
class SelfCrossModel(nn.Module):
    def __init__(self, bert_embedding_dim, audio_embedding_dim, lstm_hidden_dim, num_classes, num_attention_layers=2):
        super(SelfCrossModel, self).__init__()
        
        # Projections
        # self.text_projection = nn.Linear(bert_embedding_dim, lstm_hidden_dim)
        # self.audio_projection = nn.Linear(audio_embedding_dim, lstm_hidden_dim)
        # Self-attention
        self.self_attention_text = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=bert_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        self.self_attention_audio = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=audio_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        # self.self_attention_audio = nn.MultiheadAttention(embed_dim=audio_embedding_dim, num_heads=8)
        # Multi-layer cross-attention
        self.cross_attention_text = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=bert_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        self.cross_attention_audio = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=audio_embedding_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(bert_embedding_dim)
        self.dropout = nn.Dropout(0.5)

        self.text_pooling = ImprovedAttentionPooling(bert_embedding_dim)
        self.audio_pooling = ImprovedAttentionPooling(audio_embedding_dim)
        # LSTM for sequential learning
        # self.lstm_self_text = nn.LSTM(input_size=bert_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        # self.lstm_self_audio = nn.LSTM(input_size=audio_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        # self.lstm_text = nn.LSTM(input_size=bert_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        # self.lstm_audio = nn.LSTM(input_size=bert_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(bert_embedding_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, bert_embeddings, audio_embeddings):
        # Projections
        # print(f'bert_embeddings INIT shape: {bert_embeddings.shape}')
        # print(f'audio_embeddings INIT shape: {audio_embeddings.shape}')
        # bert_embeddings = self.text_projection(bert_embeddings)
        # audio_embeddings = self.audio_projection(audio_embeddings)
        # print(f'bert_embeddings PROJECTION shape: {bert_embeddings.shape}')
        # print(f'audio_embeddings PROJECTION shape: {audio_embeddings.shape}')
        bert_embeddings = bert_embeddings.permute(1, 0, 2)  # (seq_len_text, batch_size, bert_embedding_dim)
        audio_embeddings = audio_embeddings.permute(1, 0, 2) 
        # print(f'bert_embeddings BEFORE ATTENTION shape: {bert_embeddings.shape}')
        # print(f'audio_embeddings BEFORE ATTENTION shape: {audio_embeddings.shape}')
        # Multi-step cross-attention
        new_bert_embeddings = bert_embeddings
        self_bert_embeddings = bert_embeddings
        self_audio_embeddings = audio_embeddings
        new_audio_embeddings = audio_embeddings
        for i in range(len(self.cross_attention_text)):
            new_bert_embeddings, _ = self.cross_attention_text[i](
                query=new_bert_embeddings, key=audio_embeddings, value=audio_embeddings)
            new_audio_embeddings, _ = self.cross_attention_audio[i](
                query=new_audio_embeddings, key=bert_embeddings, value=bert_embeddings)
            self_bert_embeddings, _ = self.self_attention_text[i](
                query=self_bert_embeddings, key=self_bert_embeddings, value=self_bert_embeddings)
            self_audio_embeddings, _ = self.self_attention_audio[i](
                query=self_audio_embeddings, key=self_audio_embeddings, value=self_audio_embeddings)
            
            # Residual + normalization
            new_bert_embeddings = self.layer_norm(new_bert_embeddings + self.dropout(new_bert_embeddings))
            new_audio_embeddings = self.layer_norm(new_audio_embeddings + self.dropout(new_audio_embeddings))
            self_bert_embeddings = self.layer_norm(self_bert_embeddings + self.dropout(self_bert_embeddings))
            self_audio_embeddings = self.layer_norm(self_audio_embeddings + self.dropout(self_audio_embeddings))
        
        # print(f'bert_embeddings AFTER ATTENTION shape: {bert_embeddings.shape}')
        # print(f'audio_embeddings AFTER ATTENTION shape: {audio_embeddings.shape}')
        text_output = new_bert_embeddings.permute(1, 0, 2)
        audio_output = new_audio_embeddings.permute(1, 0, 2)
        self_text_output = self_bert_embeddings.permute(1, 0, 2)
        self_audio_output = self_audio_embeddings.permute(1, 0, 2)
        # # Attention Pooling
        # text_output_mean = text_output.mean(dim=1)
        # text_output_max = text_output.max(dim=1).values
        # audio_output_mean = audio_output.mean(dim=1)
        # audio_output_max = audio_output.max(dim=1).values
        # self_text_output_mean = self_text_output.mean(dim=1)
        # self_text_output_max = self_text_output.max(dim=1).values
        # self_audio_output_mean = self_audio_output.mean(dim=1)
        # self_audio_output_max = self_audio_output.max(dim=1).values
        
        text_output = self.text_pooling(text_output)
        audio_output = self.audio_pooling(audio_output)
        self_text_output = self.text_pooling(self_text_output)
        self_audio_output = self.audio_pooling(self_audio_output)
        
        
        # LSTM processing
        # print(f'text_output BEFORE LSTM shape: {text_output.shape}')
        # print(f'audio_output BEFORE LSTM shape: {audio_output.shape}')
        # text_output, _ = self.lstm_text(text_output)
        # audio_output, _ = self.lstm_audio(audio_output)
        # self_text_output, _ = self.lstm_self_text(self_text_output)
        # self_audio_output, _ = self.lstm_self_audio(self_audio_output)
        

        # Concatenate last hidden states
        # text_last = text_output[:, -1, :]
        # audio_last = audio_output[:, -1, :]
        # self_text_last = self_text_output[:, -1, :]
        # self_audio_last = self_audio_output[:, -1, :]
        combined_output = torch.cat((text_output, audio_output,self_text_output,self_audio_output), dim=1)
        
        # Classification
        logits = self.fc(combined_output)
        probabilities = self.softmax(logits)

        return probabilities