import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from decoder import Seq2Seq, Encoder, Decoder, input_vocab_to_index, output_vocab_to_index, input_vocab_size, output_vocab_size, input_index_to_vocab, output_index_to_vocab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameters
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 128
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 20
LEARNING_RATE = 0.001

# Load dataset
csv_filename = 'encoded_korean_texts.csv'
data = pd.read_csv(csv_filename, quotechar='"', escapechar='\\', encoding='utf-8', on_bad_lines='skip')
data = data[data.columns[:2]]
data.columns = ['encoded_text', 'original_text']

# Preprocess data
train_data = data[:-2000]
val_data = data[-2000:]

# Preprocessing function
def preprocess_data(data, input_vocab_to_index, output_vocab_to_index, max_length=100):
    input_data, output_data = [], []
    for _, row in data.iterrows():
        input_text = row['encoded_text']
        output_text = row['original_text']
        input_indices = [input_vocab_to_index.get(char, input_vocab_to_index['<PAD>']) for char in input_text]
        input_indices = input_indices[:max_length] + [input_vocab_to_index['<PAD>']] * (max_length - len(input_indices))
        output_indices = [output_vocab_to_index['<SOS>']] + \
                         [output_vocab_to_index.get(char, output_vocab_to_index['<PAD>']) for char in output_text] + \
                         [output_vocab_to_index['<EOS>']]
        output_indices = output_indices[:max_length] + [output_vocab_to_index['<PAD>']] * (max_length - len(output_indices))
        input_data.append(input_indices)
        output_data.append(output_indices)
    return torch.tensor(input_data), torch.tensor(output_data)

train_input, train_output = preprocess_data(train_data, input_vocab_to_index, output_vocab_to_index)
val_input, val_output = preprocess_data(val_data, input_vocab_to_index, output_vocab_to_index)

# Data loaders
batch_size = 32
train_loader = DataLoader(list(zip(train_input, train_output)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(list(zip(val_input, val_output)), batch_size=batch_size, shuffle=False)

# Initialize model components
encoder = Encoder(input_vocab_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(output_vocab_size, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=output_vocab_to_index['<PAD>'])
smoothing_function = SmoothingFunction().method1

print("Model initialized and ready for training")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predictions = output.argmax(dim=1)
        train_correct += (predictions == trg).sum().item()
        train_total += trg.numel()
    train_accuracy = train_correct / train_total * 100
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

# Validation only in the last epoch
if epoch == EPOCHS - 1:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    bleu_scores = []
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            val_loss += loss.item()
            predictions = output.argmax(dim=1)
            val_correct += (predictions == trg).sum().item()
            val_total += trg.numel()
            for pred, target in zip(predictions, trg):
                pred_text = ''.join([output_index_to_vocab[idx.item()] for idx in [pred] if idx.item() != output_vocab_to_index['<PAD>']])
                target_text = ''.join([output_index_to_vocab[idx.item()] for idx in [target] if idx.item() != output_vocab_to_index['<PAD>']])
                reference = [list(target_text)]
                hypothesis = list(pred_text)
                bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
                bleu_scores.append(bleu)
    val_accuracy = val_correct / val_total * 100
    avg_val_loss = val_loss / len(val_loader)
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, BLEU Score: {avg_bleu_score:.4f}")

print("Training completed!")
