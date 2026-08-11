import os
import sys
import json
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharacterTokenizer
from model import CharSeq2SeqTransformer

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
PARALLEL_CORPUS = os.path.join(DATA_DIR, "parallel_corpus.jsonl")
TOKENIZER_VOCAB = os.path.join(DATA_DIR, "tokenizer_char_vocab.json")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "model_roman_toto.pt")

class ParallelCorpusRomanDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        eng_text = pair["english"]
        toto_text = pair["toto_roman"]

        src_ids = self.tokenizer.encode(eng_text, add_bos=True, add_eos=True)[:self.max_len]
        tgt_ids = self.tokenizer.encode(toto_text, add_bos=True, add_eos=True)[:self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch, pad_idx=0):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded

def load_dataset(corpus_path):
    pairs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("english") and item.get("toto_roman"):
                    pairs.append(item)
    return pairs

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(val_loader))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharacterTokenizer.load(TOKENIZER_VOCAB)
    print(f"Loaded tokenizer with {tokenizer.vocab_size} character tokens.")

    pairs = load_dataset(PARALLEL_CORPUS)
    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.85)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Dataset split (Romanized Toto): {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs.")

    train_dataset = ParallelCorpusRomanDataset(train_pairs, tokenizer)
    val_dataset = ParallelCorpusRomanDataset(val_pairs, tokenizer)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    model = CharSeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=tokenizer.pad_id
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 20

    best_val_loss = float("inf")
    print("\n--- STARTING ROMANIZED TOTO TRAINING ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved_str = "[SAVED BEST]"
        else:
            saved_str = ""

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {saved_str}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds. Best model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
