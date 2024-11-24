
from torch.utils.data import Dataset
import torch

class MELDDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
        - dataframe: Un DataFrame que contiene las columnas text_embeddings, audio_embeddings y emotion.
        """
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text_embedding = torch.tensor(self.dataframe.iloc[idx]["text_embeddings"], dtype=torch.float32)
        audio_embedding = torch.tensor(self.dataframe.iloc[idx]["audio_embeddings"], dtype=torch.float32)
        label = torch.tensor(self.dataframe.iloc[idx]["emotion"], dtype=torch.long)
        return text_embedding, audio_embedding, label