import torch
from torch.utils.data import Dataset

class KoreanObfuscationDataset(Dataset):
    def __init__(self, input_file, target_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(input_file, encoding='utf-8') as f:
            self.inputs = [line.strip() for line in f if line.strip()]

        with open(target_file, encoding='utf-8') as f:
            self.targets = [line.strip() for line in f if line.strip()]

        assert len(self.inputs) == len(self.targets), "Mismatched line counts"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length)
        label_ids = self.tokenizer.encode(target_text, max_length=self.max_length)

        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

