import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
PARALLEL_CORPUS = os.path.join(DATA_DIR, "parallel_corpus.jsonl")
TOKENIZER_VOCAB = os.path.join(DATA_DIR, "tokenizer_char_vocab.json")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]

class CharacterTokenizer:
    def __init__(self, vocab=None):
        if vocab:
            self.char2idx = vocab
            self.idx2char = {int(idx): char for char, idx in vocab.items()}
        else:
            self.char2idx = {}
            self.idx2char = {}

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def pad_id(self):
        return self.char2idx["<pad>"]

    @property
    def unk_id(self):
        return self.char2idx["<unk>"]

    @property
    def bos_id(self):
        return self.char2idx["<bos>"]

    @property
    def eos_id(self):
        return self.char2idx["<eos>"]

    @property
    def sep_id(self):
        return self.char2idx["<sep>"]

    def build_vocab(self, corpus_path):
        unique_chars = set()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                eng = item.get("english", "")
                toto_ben = item.get("toto_bengali", "")
                toto_rom = item.get("toto_roman", "")
                
                unique_chars.update(eng)
                unique_chars.update(toto_ben)
                unique_chars.update(toto_rom)

        # Sort characters for deterministic ID assignment
        sorted_chars = sorted(list(unique_chars))

        self.char2idx = {}
        for token in SPECIAL_TOKENS:
            self.char2idx[token] = len(self.char2idx)

        for char in sorted_chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)

        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        print(f"Built character vocabulary with {len(self.char2idx)} total tokens.")

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for char in text:
            ids.append(self.char2idx.get(char, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id, self.sep_id):
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, filepath=TOKENIZER_VOCAB):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)
        print(f"Saved character vocabulary to {filepath}")

    @classmethod
    def load(cls, filepath=TOKENIZER_VOCAB):
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


def main():
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(PARALLEL_CORPUS)
    tokenizer.save(TOKENIZER_VOCAB)

    # Sanity test
    test_eng = "The sky is blue."
    test_toto = "দিংবা হা ইউইনি মি"
    
    encoded_eng = tokenizer.encode(test_eng, add_bos=True)
    encoded_toto = tokenizer.encode(test_toto, add_eos=True)
    
    print("\n--- SANITY TEST ---")
    print(f"Original Eng : {test_eng}")
    print(f"Encoded Eng  : {encoded_eng}")
    print(f"Decoded Eng  : {tokenizer.decode(encoded_eng)}")
    
    print(f"Original Toto: {test_toto}")
    print(f"Encoded Toto : {encoded_toto}")
    print(f"Decoded Toto : {tokenizer.decode(encoded_toto)}")


if __name__ == "__main__":
    main()
