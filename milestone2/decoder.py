import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from jamotools import split_syllables, join_jamos

# Load the dataset
csv_filename = 'encoded_korean_texts.csv'

# Load and clean the dataset, dropping rows with more than 2 columns
data = pd.read_csv(csv_filename)

# Drop rows with more than 2 columns
data = data[data.columns[:2]]

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=2000, random_state=42)

# Character-level tokenization with Jamo conversion
def build_vocab(texts):
    vocab = set()
    for text in texts:
        jamo_text = split_syllables(text)  # Convert Hangul to Jamo
        vocab.update(list(jamo_text))
    vocab = sorted(list(vocab))
    vocab_to_index = {char: idx for idx, char in enumerate(vocab)}
    index_to_vocab = {idx: char for char, idx in vocab_to_index.items()}
    return vocab_to_index, index_to_vocab, len(vocab)

input_vocab_to_index, input_index_to_vocab, input_vocab_size = build_vocab(data['encoded_text'])
output_vocab_to_index, output_index_to_vocab, output_vocab_size = build_vocab(data['original_text'])

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

for token in special_tokens:
    input_vocab_to_index[token] = len(input_vocab_to_index)
    output_vocab_to_index[token] = len(output_vocab_to_index)
    input_index_to_vocab[len(input_index_to_vocab)] = token
    output_index_to_vocab[len(output_index_to_vocab)] = token

input_vocab_size += len(special_tokens)
output_vocab_size += len(special_tokens)

# Prepare training and validation tensors
def preprocess_data(data, input_vocab_to_index, output_vocab_to_index, max_length=100):
    input_data, output_data = [], []
    for _, row in data.iterrows():
        input_text = split_syllables(row['encoded_text'])
        output_text = split_syllables(row['original_text'])

        input_indices = [input_vocab_to_index.get(char, input_vocab_to_index[PAD_TOKEN]) for char in input_text]
        input_indices = input_indices[:max_length] + [input_vocab_to_index[PAD_TOKEN]] * (max_length - len(input_indices))

        output_indices = [output_vocab_to_index[SOS_TOKEN]] + \
                         [output_vocab_to_index.get(char, output_vocab_to_index[PAD_TOKEN]) for char in output_text] + \
                         [output_vocab_to_index[EOS_TOKEN]]
        output_indices = output_indices[:max_length] + [output_vocab_to_index[PAD_TOKEN]] * (max_length - len(output_indices))

        input_data.append(input_indices)
        output_data.append(output_indices)
    return torch.tensor(input_data), torch.tensor(output_data)

# Define the Encoder, Decoder, and Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1

        return outputs

print("Model initialized and ready for training")
