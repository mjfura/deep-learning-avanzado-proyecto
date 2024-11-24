import torch
import torchaudio
from transformers import BertTokenizer, BertModel

class EmbeddingModel():
    def __init__(self):
        self.tokenizer_text = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_text = BertModel.from_pretrained('bert-base-uncased')
        self.bundle_audio = torchaudio.pipelines.WAV2VEC2_BASE
        self.model_audio = self.bundle_audio.get_model()

    def embed_text(self, sentence:str):
        inputs = self.tokenizer_text(sentence, return_tensors='pt')
    
        # Get the embeddings
        with torch.no_grad():
            print(f'Input type: {type(inputs)}')
            print(f'Inputs Shape: {inputs["input_ids"].shape}')
            outputs = self.model_text(**inputs)
        
        # The last hidden state is the output of the model
        embeddings = outputs.last_hidden_state
        return embeddings
    def embed_audio(self,wav_file:str):
        waveform, sample_rate = torchaudio.load(wav_file)
        print(f'Sample rate: {sample_rate}')
        print(f'Waveform shape before: {waveform.shape}')
        # Ensure the audio is mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        print(f'Bundle sample rate: {self.bundle_audio.sample_rate}')
        if sample_rate != self.bundle_audio.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.bundle_audio.sample_rate)(waveform)
        
        # Get the embeddings
        with torch.inference_mode():
            print(f'Waveform shape: {waveform.shape}')
            embeddings = self.model_audio(waveform)
        
        return embeddings[0]