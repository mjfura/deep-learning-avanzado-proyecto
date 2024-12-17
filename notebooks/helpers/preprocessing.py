import os
from pydub import AudioSegment
from .embeddings import EmbeddingModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
class PreprocessingDataframe:
    def __init__(self, dataframe,folder_audio_path:str="../datasets/MELD.Raw/dev_splits_complete"):
        self.df = dataframe
        self.label_encoder = LabelEncoder()
        self.folder_path = folder_audio_path
    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)
        return self

    def drop_na(self):
        self.df = self.df.dropna()
        self.df=self.df.dropna(subset=['Emotion'])
        self.df=self.df.dropna(subset=['wav_file'])
        return self

    def fill_na(self, value):
        self.df = self.df.fillna(value)
        return self

    def get_dataframe(self):
        return self.df
    
    def extract_audio(self,mp4_file, wav_file):
        if not os.path.isfile(mp4_file):
            raise FileNotFoundError(f"The file {mp4_file} does not exist.")
        audio = AudioSegment.from_file(mp4_file, format="mp4")
        audio.export(wav_file, format="wav")
    
    def set_path_wavfiles(self):
        for i in range(self.df.shape[0]):
            row_data = self.df.iloc[i]
            wav_file_path = f"{self.folder_path}/dia{row_data['Dialogue_ID']}_utt{row_data['Utterance_ID']}.wav"
            if os.path.isfile(wav_file_path):
                self.df.at[i, "wav_file"] = wav_file_path
        return self
    
    def generate_wavfiles_from_df(self):
        have_to_create = True
        if 'wav_file' in self.df.columns:
            print("La columna 'wav_file' existe en el DataFrame.")
            have_to_create = False
        for i in range(self.df.shape[0]):
            row_data = self.df.iloc[i]
            mp4_file = f"{self.folder_path}/dia{row_data['Dialogue_ID']}_utt{row_data['Utterance_ID']}.mp4"
            wav_file = f"{self.folder_path}/dia{row_data['Dialogue_ID']}_utt{row_data['Utterance_ID']}.wav"
            if not have_to_create:
                if type(row_data["wav_file"]) is str:
                    print(f"Skipping row {i} {type(row_data['wav_file'])}")
                else:
                    try:
                        self.extract_audio(mp4_file, wav_file)
                        print(f"Extracted audio for row {i}")
                        self.df.at[i, "wav_file"] = wav_file
                    except Exception as e:
                        print(f"Error extracting audio for row {i} {e}")
            else:
                try:
                    self.extract_audio(mp4_file, wav_file)
                    print(f"Extracted audio for row {i}")
                    self.df.at[i, "wav_file"] = wav_file
                except Exception as e:
                    print(f"Error extracting audio for row {i} {e}")
    def add_embeddings(self,embedding_model:EmbeddingModel):
        self.df["audio_embeddings"] = None
        self.df["text_embeddings"] = None
        for i in range(self.df.shape[0]):
            row_data = self.df.iloc[i]
            if type(row_data["wav_file"]) is str:
                try:
                    print('Generating audio embeddings for ', row_data["wav_file"])
                    embeddings = embedding_model.embed_audio(row_data["wav_file"])
                    print('before saving in DF',type(embeddings))
                    self.df.at[i, "audio_embeddings"] = embeddings
                except Exception as e:
                    print(f"Error extracting audio for row {i} {e}")
            print('Generating text embeddings for ', row_data["Utterance"])
            embedding_texto = embedding_model.embed_text(row_data["Utterance"])
            print('before saving in DF ',type(embedding_texto))
            self.df.at[i, "text_embeddings"] = embedding_texto
        return self
    
    def add_label_encoder_to_emotion(self):
        self.df["emotion"] = self.label_encoder.fit_transform(self.df["Emotion"])
        return self
    
    def save_embeddings(self, path):
        self.df.to_pickle(path)
        return self
    
    def load_embeddings(self, path):
        self.df = pd.read_pickle(path)
        return self
    def add_clap_embeddings(self,embedding_model:EmbeddingModel):
        self.df["audio_embeddings"] = None
        self.df["text_embeddings"] = None
        for i in range(self.df.shape[0]):
            row_data = self.df.iloc[i]
            if type(row_data["wav_file"]) is str:
                try:
                    print('Generating audio embeddings for ', row_data["wav_file"])
                    embeddings = embedding_model.embed_audio(row_data["wav_file"])
                    print('before saving in DF',type(embeddings))
                    self.df.at[i, "audio_embeddings"] = embeddings
                except Exception as e:
                    print(f"Error extracting audio for row {i} {e}")
            print('Generating text embeddings for ', row_data["Utterance"])
            embedding_texto = embedding_model.embed_text(row_data["Emotion"])
            print('before saving in DF ',type(embedding_texto))
            self.df.at[i, "text_embeddings"] = embedding_texto
        return self