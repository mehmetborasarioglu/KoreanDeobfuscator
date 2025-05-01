import torch
print(1)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartConfig
print(2)
from datasets import Dataset
import pandas as pd
from g2pk import G2p
import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('cache', exist_ok=True)

def load_data(max_samples=1000000, cache_dir='cache'):
    phonetic_cache_file = os.path.join(cache_dir, f'phonetic_cache_{max_samples}.pkl')
    
    if os.path.exists(phonetic_cache_file):
        print(f"Loading cached phonetic data from {phonetic_cache_file}...")
        with open(phonetic_cache_file, 'rb') as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    
    print("No cached data found. Processing from scratch...")
    
    with open('data/original.txt', 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f.readlines()]
    
    with open('data/obfuscated.txt', 'r', encoding='utf-8') as f:
        obfuscated_lines = [line.strip() for line in f.readlines()]
    
    if len(original_lines) > max_samples:
        original_lines = original_lines[:max_samples]
        obfuscated_lines = obfuscated_lines[:max_samples]
    
    g2p = G2p()
    print("Converting to phonetic representations...")
    
    phonetic_original = []
    skipped_indices = []
    
    for i, text in enumerate(tqdm(original_lines, desc="Converting original texts")):
        try:
            phonetic_original.append(g2p(text))
        except Exception as e:
            print(f"Error processing line {i} of original texts: {e}")
            print(f"Problematic text: {text}")
            phonetic_original.append(text)  # Use original text as fallback
            skipped_indices.append(i)
    
    phonetic_obfuscated = []
    for i, text in enumerate(tqdm(obfuscated_lines, desc="Converting obfuscated texts")):
        if i in skipped_indices:
            phonetic_obfuscated.append(text)
            continue
            
        try:
            phonetic_obfuscated.append(g2p(text))
        except Exception as e:
            print(f"Error processing line {i} of obfuscated texts: {e}")
            print(f"Problematic text: {text}")
            phonetic_obfuscated.append(text)

            if i not in skipped_indices:
                phonetic_original[i] = original_lines[i]

    data = {
        'original': original_lines,
        'obfuscated': obfuscated_lines,
        'phonetic_original': phonetic_original,
        'phonetic_obfuscated': phonetic_obfuscated
    }
    
    # Save to cache after each 100,000 samples processed
    if len(phonetic_original) >= 100000:
        temp_cache_file = os.path.join(cache_dir, f'phonetic_cache_temp_{len(phonetic_original)}.pkl')
        print(f"Saving intermediate phonetic data to cache: {temp_cache_file}")
        with open(temp_cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    # Save final results to cache
    print(f"Saving phonetic data to cache: {phonetic_cache_file}")
    with open(phonetic_cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return pd.DataFrame(data)

class CharTokenizer:
    def __init__(self, char_to_id, id_to_char):
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
    
    def tokenize(self, text):
        return [self.char_to_id.get(c, self.char_to_id['<unk>']) for c in text]
    
    def decode(self, ids):
        return ''.join([self.id_to_char.get(id, '<unk>') for id in ids if id not in [0, 2, 3]])  # Skip special tokens

def create_char_tokenizer(texts, cache_dir='cache'):
    tokenizer_cache_file = os.path.join(cache_dir, 'char_tokenizer.pkl')
    
    if os.path.exists(tokenizer_cache_file):
        print(f"Loading cached tokenizer from {tokenizer_cache_file}...")
        with open(tokenizer_cache_file, 'rb') as f:
            tokenizer_data = pickle.load(f)
        tokenizer = tokenizer_data['tokenizer']
        return tokenizer.tokenize, tokenizer.decode, tokenizer_data['vocab_size']

    print("No cached tokenizer found. Creating new tokenizer...")
    
    print("Building character vocabulary...")
    unique_chars = set()
    for text in tqdm(texts, desc="Processing texts for tokenizer"):
        unique_chars.update(text)
    
    vocab = ['<pad>', '<unk>', '<s>', '</s>'] + sorted(list(unique_chars))
    
    char_to_id = {char: i for i, char in enumerate(vocab)}
    id_to_char = {i: char for i, char in enumerate(vocab)}
    
    tokenizer = CharTokenizer(char_to_id, id_to_char)
    
    tokenizer_data = {
        'tokenizer': tokenizer,
        'vocab_size': len(vocab),
        'vocab': vocab
    }
    
    print(f"Saving tokenizer to cache: {tokenizer_cache_file}")
    with open(tokenizer_cache_file, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    
    return tokenizer.tokenize, tokenizer.decode, len(vocab)

def prepare_datasets(df, cache_dir='cache'):
    datasets_cache_file = os.path.join(cache_dir, f'datasets_{len(df)}.pkl')
    
    if os.path.exists(datasets_cache_file):
        print(f"Loading cached datasets from {datasets_cache_file}...")
        with open(datasets_cache_file, 'rb') as f:
            datasets = pickle.load(f)
        return datasets['train'], datasets['val']
    
    print("No cached datasets found. Preparing from scratch...")
    
    print("Splitting into train/validation sets...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    print(f"Saving datasets to cache: {datasets_cache_file}")
    with open(datasets_cache_file, 'wb') as f:
        pickle.dump(datasets, f)
    
    return train_dataset, val_dataset

class CustomDataCollator:
    def __init__(self, tokenize_fn, max_length=128, pad_token_id=0):
        self.tokenize_fn = tokenize_fn
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples):
        input_ids_list = []
        attention_masks = []
        labels_list = []
        
        for example in examples:
            input_tokens = self.tokenize_fn(example["phonetic_obfuscated"])
            
            attention_mask = [1] * len(input_tokens)
            
            if len(input_tokens) > self.max_length:
                input_tokens = input_tokens[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            else:
                padding_length = self.max_length - len(input_tokens)
                input_tokens = input_tokens + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            labels = self.tokenize_fn(example["phonetic_original"])
            
            if len(labels) > self.max_length:
                labels = labels[:self.max_length]
            else:
                padding_length = self.max_length - len(labels)
                labels = labels + [self.pad_token_id] * padding_length
            
            input_ids_list.append(input_tokens)
            attention_masks.append(attention_mask)
            labels_list.append(labels)
        
        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        
        return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(-1)
    
    labels_ids[labels_ids == -100] = 0
    
    correct = 0
    total = 0
    
    for pred_seq, label_seq in zip(pred_ids, labels_ids):

        pred_tokens = [token for token in pred_seq if token != 0]
        label_tokens = [token for token in label_seq if token != 0]
        
        if pred_tokens == label_tokens:
            correct += 1
        total += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0,
    }

def main():

    use_cache = True
    cache_dir = 'cache'
    
    import transformers
    
    print("Loading data...")
    df = load_data(max_samples=1000000, cache_dir=cache_dir if use_cache else None)
    
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(df, cache_dir=cache_dir if use_cache else None)

    print("Creating tokenizer...")
    all_texts = df['phonetic_obfuscated'].tolist() + df['phonetic_original'].tolist()
    tokenize_fn, decode_fn, vocab_size = create_char_tokenizer(all_texts, cache_dir=cache_dir if use_cache else None)
    
    print("Initializing model from scratch with custom vocabulary...")
    
    config = BartConfig.from_pretrained(
        "facebook/bart-base", 
        vocab_size=vocab_size,
        d_model=768, 
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        decoder_ffn_dim=3072,
        encoder_ffn_dim=3072,
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        forced_bos_token_id=2,
        forced_eos_token_id=3,
    )
    
    from transformers import BartForConditionalGeneration
    model = BartForConditionalGeneration(config)
    print(f"Model initialized with vocabulary size: {vocab_size}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch", 
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,  
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=100,  
        logging_first_step=True,
        remove_unused_columns=False, 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )
    
    data_collator = CustomDataCollator(tokenize_fn, max_length=128)
    
    overall_progress = tqdm(
        total=int(training_args.num_train_epochs * (len(train_dataset) // training_args.per_device_train_batch_size)),
        desc="Training progress",
        position=0
    )
    
    class TqdmCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            overall_progress.update(1)
            if state.log_history and len(state.log_history) > 0 and 'loss' in state.log_history[-1]:
                overall_progress.set_description(f"Training (loss: {state.log_history[-1]['loss']:.4f})")
            
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and 'eval_accuracy' in metrics:
                overall_progress.set_postfix(eval_acc=metrics['eval_accuracy'])
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        callbacks=[TqdmCallback()],
    )
    
    print("Training model...")
    trainer.train()
    overall_progress.close() 
    
    print("Saving model...")
    trainer.save_model("./checkpoints/final_model")
    
    print("\nTesting deobfuscation on sample texts:")
    test_examples = val_dataset.select(range(10)) 
    
    for i, example in enumerate(tqdm(test_examples, desc="Testing samples")):
        obfuscated = example['phonetic_obfuscated']
        original = example['phonetic_original']
        
        input_ids = torch.tensor([tokenize_fn(obfuscated)]).to(model.device)
        
        outputs = model.generate(
            input_ids,
            max_length=min(128, len(obfuscated) + 10), 
            early_stopping=True,
            no_repeat_ngram_size=3, 
            num_beams=5, 
            length_penalty=1.0  
        )
        predicted = decode_fn(outputs[0].tolist())
        
        if '.' in predicted:
            last_period_idx = predicted.rfind('.')
            if last_period_idx > len(predicted) * 0.5: 
                predicted = predicted[:last_period_idx+1]
        
        print(f"Sample {i+1}:")
        print(f"Obfuscated: {obfuscated}")
        print(f"Original:   {original}")
        print(f"Predicted:  {predicted}")
        print("-" * 50)

if __name__ == "__main__":
    main()