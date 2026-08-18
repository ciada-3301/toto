import os
import sys
import torch
from tokenizer import CharacterTokenizer
from model import CharSeq2SeqTransformer

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
TOKENIZER_VOCAB = os.path.join(DATA_DIR, "tokenizer_char_vocab.json")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "model_bengali_toto.pt")

def translate(english_text, model, tokenizer, device, max_len=128):
    model.eval()
    src_ids = tokenizer.encode(english_text, add_bos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    tgt_ids = [tokenizer.bos_id]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            output = model(src_tensor, tgt_tensor)
            next_token_id = output[0, -1, :].argmax(dim=-1).item()

            if next_token_id == tokenizer.eos_id:
                break
            tgt_ids.append(next_token_id)

    return tokenizer.decode(tgt_ids)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}. Please run train.py first.")
        return

    tokenizer = CharacterTokenizer.load(TOKENIZER_VOCAB)
    model = CharSeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.15,
        pad_idx=tokenizer.pad_id
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Loaded trained English -> Bengali Toto character model.")

    test_sentences = [
        "The sky is blue.",
        "The cat is sleeping.",
        "Water",
        "Come on, let’s go fishing in the river today.",
        "My child",
        "The elephant is drinking water from the river",
        "Sun",
        "Bright"
    ]

    print("\n================ TRANSLATION INFERENCE RESULTS ================")
    for eng in test_sentences:
        toto_pred = translate(eng, model, tokenizer, device)
        print(f"English: {eng}")
        print(f"Toto   : {toto_pred}\n")

if __name__ == "__main__":
    main()
