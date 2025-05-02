# Import section at the top of the file
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jamo import h2j, j2h
from tqdm import tqdm
import os

class PositionalEncoding(nn.Module):
    """Adds positional information to token embeddings using sine and cosine functions."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SyllableComposer(nn.Module):
    """Composes jamo features into syllable representations."""
    def __init__(self, d_model):
        super().__init__()
        self.initial_projector = nn.Linear(d_model, d_model)
        self.medial_projector = nn.Linear(d_model, d_model)
        self.final_projector = nn.Linear(d_model, d_model)
        self.combiner = nn.Linear(d_model * 3, d_model)
        self.gate = nn.Linear(d_model * 3, 3)
        
    def forward(self, jamo_features, jamo_positions):
        batch_size, seq_len, hidden_dim = jamo_features.shape
        
        position_probs = F.softmax(jamo_positions, dim=-1)
        
        initial_weighted = jamo_features * position_probs[:, :, 0:1]
        medial_weighted = jamo_features * position_probs[:, :, 1:2]
        final_weighted = jamo_features * position_probs[:, :, 2:3]
        
        initial_features = self.initial_projector(initial_weighted)
        medial_features = self.medial_projector(medial_weighted)
        final_features = self.final_projector(final_weighted)
        
        syllable_starts = (position_probs[:, :, 0] > 0.5).float()
        
        syllable_indices = torch.cumsum(syllable_starts, dim=1)
        syllable_indices = syllable_indices * (1 - (jamo_positions.sum(dim=-1) == 0).float())
        
        max_syllables = torch.max(syllable_indices, dim=1)[0].int() + 1
        max_syllable_count = max_syllables.max().item()

        syllable_features = torch.zeros(batch_size, max_syllable_count, hidden_dim, device=jamo_features.device)
        syllable_boundaries = []
        
        for b in range(batch_size):
            b_boundaries = []

            for s in range(1, max_syllables[b].item() + 1):
                syllable_mask = (syllable_indices[b] == s).float().unsqueeze(-1)
                
                syl_initial = (initial_features[b] * syllable_mask).sum(dim=0)
                syl_medial = (medial_features[b] * syllable_mask).sum(dim=0)
                syl_final = (final_features[b] * syllable_mask).sum(dim=0)
                
                pos_mask = (syllable_indices[b] == s)
                positions = pos_mask.nonzero().squeeze(-1)
                if positions.numel() > 0:
                    start = positions.min().item()
                    end = positions.max().item() + 1
                    b_boundaries.append((start, end))
                
                combined = torch.cat([syl_initial, syl_medial, syl_final], dim=-1)
                
                gates = F.softmax(self.gate(combined), dim=-1)
                
                syl_initial = syl_initial * gates[0]
                syl_medial = syl_medial * gates[1]
                syl_final = syl_final * gates[2]

                combined = torch.cat([syl_initial, syl_medial, syl_final], dim=-1)
                syllable_features[b, s-1] = self.combiner(combined)
            
            syllable_boundaries.append(b_boundaries)
                
        return syllable_features, syllable_boundaries

class JamoPositionModule(nn.Module):
    """Predicts the position type of each jamo (initial, medial, final)."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_model // 2, 3)
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.dense2(x)

class CrossLevelAttention(nn.Module):
    """Enables information flow between jamo and syllable levels."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        self.syllable_to_jamo = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.jamo_to_syllable = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, jamo_features, syllable_features, syllable_indices):
        batch_size, seq_len, d_model = jamo_features.shape

        enhanced_jamo = jamo_features.clone()
        
        syllable_context = self.syllable_to_jamo(syllable_features)

        for b in range(batch_size):
            for j in range(seq_len):
                syl_idx = int(syllable_indices[b, j].item()) - 1
                if syl_idx >= 0 and syl_idx < syllable_features.size(1):
                    enhanced_jamo[b, j] += syllable_context[b, syl_idx]

        enhanced_syllable = syllable_features.clone()
        jamo_context = self.jamo_to_syllable(jamo_features)

        for b in range(batch_size):
            syllable_counts = torch.zeros(syllable_features.size(1), device=jamo_features.device)
            syllable_sum = torch.zeros(syllable_features.size(1), d_model, device=jamo_features.device)
            
            for j in range(seq_len):
                syl_idx = int(syllable_indices[b, j].item()) - 1
                if syl_idx >= 0 and syl_idx < syllable_features.size(1):
                    syllable_sum[syl_idx] += jamo_context[b, j]
                    syllable_counts[syl_idx] += 1

            for s in range(syllable_features.size(1)):
                if syllable_counts[s] > 0:
                    enhanced_syllable[b, s] += syllable_sum[s] / syllable_counts[s]

        jamo_output = self.norm1(enhanced_jamo)
        syllable_output = self.norm2(enhanced_syllable)
        
        return jamo_output, syllable_output

class HierarchicalEncoder(nn.Module):
    """Encoder that processes information at both jamo and syllable levels."""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout=0.1
    ):
        super().__init__()

        self.position_predictor = JamoPositionModule(d_model, dropout)

        self.syllable_composer = SyllableComposer(d_model)

        self.jamo_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.syllable_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cross_attention = nn.ModuleList([
            CrossLevelAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, src_mask=None):
        jamo_positions = self.position_predictor(x)
        
        syllable_features, syllable_boundaries = self.syllable_composer(x, jamo_positions)
        
        batch_size, seq_len, _ = x.shape
        syllable_indices = torch.zeros(batch_size, seq_len, device=x.device)
        
        for b in range(batch_size):
            for s, (start, end) in enumerate(syllable_boundaries[b]):
                syllable_indices[b, start:end] = s + 1

        jamo_features = x
        for i in range(len(self.jamo_layers)):
            if src_mask is not None:
                jamo_features = self.jamo_layers[i](jamo_features, src_key_padding_mask=src_mask)
            else:
                jamo_features = self.jamo_layers[i](jamo_features)

            syllable_features = self.syllable_layers[i](syllable_features)
            
            jamo_features, syllable_features = self.cross_attention[i](
                jamo_features, syllable_features, syllable_indices
            )
            
            jamo_positions = self.position_predictor(jamo_features)
        
        return jamo_features, syllable_features, jamo_positions, syllable_indices

class HierarchicalDecoder(nn.Module):
    """Decoder that generates outputs using both jamo and syllable-level information."""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout=0.1
    ):
        super().__init__()
        
        # Jamo-level layers
        self.jamo_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.syllable_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.position_predictor = JamoPositionModule(d_model, dropout)
        
        self.position_gate = nn.Linear(d_model, 3)
        
    def forward(self, tgt, jamo_memory, syllable_memory, syllable_indices, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        
        for i in range(len(self.jamo_layers)):
            output = self.jamo_layers[i](
                output, 
                jamo_memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            batch_size, tgt_len, _ = output.shape
            _, max_syllables, _ = syllable_memory.shape

            syllable_context, _ = self.syllable_attention[i](
                query=output, 
                key=syllable_memory, 
                value=syllable_memory
            )
            
            gate = torch.sigmoid(self.position_gate(output))
            output = output * (1 - gate[:, :, 0:1]) + syllable_context * gate[:, :, 0:1]
        
        positions = self.position_predictor(output)
        
        return output, positions

class HierarchicalJamoTransformer(nn.Module):
    """
    Hierarchical Jamo-Syllable Transformer (HJST) for Korean text deobfuscation.
    
    This model processes Korean text at both the jamo (character component) and
    syllable levels simultaneously, with information flow between levels.
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        
        self.jamo_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_encoding = PositionalEncoding(d_model, dropout)
        
        self.encoder = HierarchicalEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        self.decoder = HierarchicalDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_src_mask(self, src):
        """Create a mask to hide padding tokens in the source sequence."""
        src_key_padding_mask = (src == self.pad_token_id)
        return src_key_padding_mask
    
    def create_tgt_mask(self, tgt):
        """Create masks for the target sequence."""
        tgt_seq_len = tgt.size(1)
        
        tgt_key_padding_mask = (tgt == self.pad_token_id)

        causal_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device),
            diagonal=1
        ).bool()
        
        return tgt_key_padding_mask, causal_mask
        
    def forward(self, src, tgt):
        """
        Forward pass through the model.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
            
        Returns:
            output: Predicted token logits (batch_size, tgt_seq_len, vocab_size)
            jamo_positions: Position predictions for each output token
        """

        src_key_padding_mask = self.create_src_mask(src)
        tgt_key_padding_mask, tgt_causal_mask = self.create_tgt_mask(tgt)
        
        src_emb = self.jamo_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.position_encoding(src_emb)
        
        tgt_emb = self.jamo_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.position_encoding(tgt_emb)
        
        tgt_emb = tgt_emb[:, :-1, :]
        if tgt.size(1) > 1: 
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        
        tgt_len = tgt_emb.size(1)
        if tgt_len > 0: 
            tgt_causal_mask = tgt_causal_mask[:tgt_len, :tgt_len]
        
        jamo_memory, syllable_memory, jamo_positions, syllable_indices = self.encoder(
            src_emb, src_key_padding_mask
        )
        
        decoder_output, output_positions = self.decoder(
            tgt_emb, 
            jamo_memory,
            syllable_memory,
            syllable_indices,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.output_projection(decoder_output)
        
        return output, output_positions
    
    def generate(self, src, max_length=128, beam_size=1, progress_callback=None):
        """
        Generate output sequence from source sequence.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            max_length: Maximum generation length
            beam_size: Beam search size (1 for greedy decoding)
            progress_callback: Optional callback function for tracking progress
            
        Returns:
            generated: Generated token IDs (batch_size, gen_seq_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        src_mask = self.create_src_mask(src)
        
        src_emb = self.jamo_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.position_encoding(src_emb)
        
        jamo_memory, syllable_memory, jamo_positions, syllable_indices = self.encoder(
            src_emb, src_mask
        )
        
        if beam_size == 1:
            decoded = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # Start with BOS token
            
            for i in range(max_length - 1):
                tgt_emb = self.jamo_embedding(decoded) * math.sqrt(self.d_model)
                tgt_emb = self.position_encoding(tgt_emb)
                
                tgt_len = decoded.size(1)
                tgt_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len, device=device),
                    diagonal=1
                ).bool()
                
                decoder_output, _ = self.decoder(
                    tgt_emb,
                    jamo_memory,
                    syllable_memory, 
                    syllable_indices,
                    tgt_mask
                )
                
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                decoded = torch.cat([decoded, next_token], dim=1)
  
                if progress_callback:
                    progress_callback(i)

                if (next_token == 2).all():  # Assuming EOS token ID is 2
                    break
            
            return decoded
        else:
            raise NotImplementedError("Beam search not yet implemented")

def korean_aware_loss(predictions, targets, position_predictions, position_targets, pad_token_id):
    """
    Custom loss function that incorporates Korean linguistic structure.
    
    Args:
        predictions: Token logits (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        position_predictions: Position logits (batch_size, seq_len, 3)
        position_targets: Target positions (batch_size, seq_len)
        pad_token_id: ID of padding token to ignore
        
    Returns:
        total_loss: Combined loss value
    """
    # Standard cross-entropy for token prediction
    token_loss = F.cross_entropy(
        predictions.reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id
    )
    
    batch_size, seq_len = targets.size()
    
    flat_position_preds = position_predictions.reshape(-1, 3)
    
    mask = (targets != pad_token_id).reshape(-1)

    position_loss = F.cross_entropy(
        flat_position_preds[mask],
        position_targets.reshape(-1)[mask]
    )
    
    return token_loss + 0.2 * position_loss

def train_epoch(model, dataloader, optimizer, criterion, device, pad_token_id):
    """
    Train the model for one epoch.
    
    Args:
        model: HierarchicalJamoTransformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameters
        criterion: Loss function
        device: Training device
        pad_token_id: ID of padding token
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        output_logits, output_positions = model(input_ids, labels)
        
        shifted_labels = labels[:, 1:]

        batch_size, seq_len = shifted_labels.size()
        position_targets = torch.zeros_like(shifted_labels)
        
        for b in range(batch_size):
            position = 0 
            for i in range(seq_len):
                if shifted_labels[b, i] == pad_token_id:
                    position_targets[b, i] = -100
                else:
                    position_targets[b, i] = position
                    position = (position + 1) % 3
        
        loss = korean_aware_loss(
            output_logits, 
            shifted_labels, 
            output_positions, 
            position_targets,
            pad_token_id
        )
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, pad_token_id):
    """
    Validate the model on a dataset.
    
    Args:
        model: HierarchicalJamoTransformer model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device for computation
        pad_token_id: ID of padding token
        
    Returns:
        val_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            output_logits, output_positions = model(input_ids, labels)
            
            shifted_labels = labels[:, 1:]
            
            batch_size, seq_len = shifted_labels.size()
            position_targets = torch.zeros_like(shifted_labels)
            
            for b in range(batch_size):
                position = 0
                for i in range(seq_len):
                    if shifted_labels[b, i] == pad_token_id:
                        position_targets[b, i] = -100
                    else:
                        position_targets[b, i] = position
                        position = (position + 1) % 3
            
            loss = korean_aware_loss(
                output_logits,
                shifted_labels,
                output_positions,
                position_targets,
                pad_token_id
            )
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_predictions(model, dataloader, tokenizer, device, max_length=128):
    """
    Generate predictions for a dataset.
    
    Args:
        model: HierarchicalJamoTransformer model
        dataloader: DataLoader for data
        tokenizer: Tokenizer for encoding/decoding
        device: Device for computation
        max_length: Maximum generation length
        
    Returns:
        predictions: List of decoded predictions
        targets: List of decoded targets
    """
    model.eval()
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc="Generating Predictions")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                sample_input = input_ids[i:i+1]
                sample_target = labels[i:i+1]
                
                with tqdm(total=max_length, desc=f"Sample {i+1}/{batch_size}", leave=False) as pbar:
                    generated = model.generate(
                        sample_input, 
                        max_length=max_length,
                        progress_callback=lambda step: pbar.update(1)
                    )
                
                pred_ids = generated[0].cpu().tolist()
                target_ids = sample_target[0].cpu().tolist()
                
                pred_text = tokenizer.decode(pred_ids)
                target_text = tokenizer.decode(target_ids)
                
                predictions.append(pred_text)
                targets.append(target_text)
                
                progress_bar.set_description(f"Generating: {len(predictions)}/{len(dataloader.dataset)}")
    
    return predictions, targets

def train_model(model, train_loader, val_loader, tokenizer, device, 
               num_epochs=10, learning_rate=0.0001, pad_token_id=0, 
               checkpoint_dir="checkpoints", resume_from=None):
    """
    Train the model for multiple epochs with checkpoint saving and loading.
    
    Args:
        model: HierarchicalJamoTransformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for encoding/decoding
        device: Device for computation
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        pad_token_id: ID of padding token
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        
    Returns:
        model: Trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = korean_aware_loss
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_bar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_token_id)
        
        val_loss = validate(model, val_loader, criterion, device, pad_token_id)
        
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "hjst_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"  Saved best model checkpoint to {best_model_path}")
        
        if epoch % 2 == 0:
            sample_inputs = next(iter(val_loader))["input_ids"][:2].to(device)
            sample_targets = next(iter(val_loader))["labels"][:2].to(device)
            
            print("\nGenerating samples...")
            with tqdm(total=128, desc="  Generating", leave=False) as pbar:
                sample_predictions = model.generate(sample_inputs, max_length=128, progress_callback=lambda step: pbar.update(1))
            
            print("\nSample predictions:")
            for i in range(len(sample_inputs)):
                input_text = tokenizer.decode(sample_inputs[i].cpu().tolist())
                target_text = tokenizer.decode(sample_targets[i].cpu().tolist())
                pred_text = tokenizer.decode(sample_predictions[i].cpu().tolist())
                
                print(f"  Input:  {input_text}")
                print(f"  Target: {target_text}")
                print(f"  Pred:   {pred_text}")
                print()
    
    best_checkpoint = torch.load(os.path.join(checkpoint_dir, "hjst_best.pth"), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {best_checkpoint['epoch']+1} with validation loss: {best_checkpoint['val_loss']:.4f}")
    
    return model

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Please install tqdm: pip install tqdm")
        exit(1)
        
    import torch
    from torch.utils.data import DataLoader
    from JamoTransformer.tokenizer import JamoTokenizer 
    from JamoTransformer.dataset import KoreanObfuscationDataset 
    
    input_path = "data/obfuscated_small.txt"
    target_path = "data/original_small.txt"
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Building tokenizer...")
    with open(input_path, encoding="utf-8") as f1, open(target_path, encoding="utf-8") as f2:
        all_lines = f1.readlines() + f2.readlines()
    tokenizer = JamoTokenizer(all_lines)
    print(f"Tokenizer built with vocabulary size: {tokenizer.vocab_size()}")
    
    print("Creating dataset...")
    dataset = KoreanObfuscationDataset(input_path, target_path, tokenizer, max_length=128)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print("Checking for CUDA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing model...")

    model = HierarchicalJamoTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        num_epochs=10,
        learning_rate=0.0001,
        pad_token_id=tokenizer.pad_token_id
    )
    
    print("Generating final predictions for validation set...")
    predictions, targets = generate_predictions(
        model=trained_model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device
    )
    
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from jiwer import cer, wer
        
        print("Calculating evaluation metrics...")
        total_bleu, total_cer, total_wer = 0.0, 0.0, 0.0
        
        for pred, target in tqdm(zip(predictions, targets), desc="Evaluating", total=len(predictions)):
            bleu_score = sentence_bleu([list(target)], list(pred))
            cer_score = cer(target, pred)
            wer_score = wer(target, pred)
            
            total_bleu += bleu_score
            total_cer += cer_score
            total_wer += wer_score
        
        avg_bleu = total_bleu / len(predictions)
        avg_cer = total_cer / len(predictions)
        avg_wer = total_wer / len(predictions)
        
        print("\nEvaluation Results:")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average WER: {avg_wer:.4f}")
        
    except ImportError:
        print("NLTK and/or jiwer not installed. Please install for metrics calculation:")
        print("pip install nltk jiwer")
    
    print("\nSample results:")
    for i in range(min(5, len(predictions))):
        print(f"Input:      {tokenizer.decode([id for id in val_loader.dataset[i]['input_ids'].tolist() if id != tokenizer.pad_token_id])}")
        print(f"Prediction: {predictions[i]}")
        print(f"Target:     {targets[i]}")
        print()
    
    print("Training and evaluation complete!")
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        num_epochs=10,
        learning_rate=0.0001,
        pad_token_id=tokenizer.pad_token_id
    )
    
    predictions, targets = generate_predictions(
        model=trained_model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device
    )
    
    print("Validation complete. Sample results:")
    for i in range(min(5, len(predictions))):
        print(f"Prediction: {predictions[i]}")
        print(f"Target:     {targets[i]}")
        print()