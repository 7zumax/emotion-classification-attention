"""
emotion_utils.py
Shared code for the Emotion Classification project.
Both the notebook AND the Gradio app import this file, so a sentence typed into the app
is cleaned and encoded EXACTLY the same way the training data was.
"""
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]   # the 6 target classes
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
MAX_LEN = 40                                                             # longest comment has 31 words


# ---------- 1) Data: GoEmotions -> 6 emotions ----------
def load_goemotions(data_dir="data"):
    """Returns train / val / test DataFrames with columns: text, fine (28-label names), emotion, label."""
    names = open(f"{data_dir}/emotions.txt", encoding="utf-8").read().split()     # 28 names, index = id
    ekman = json.load(open(f"{data_dir}/ekman_mapping.json", encoding="utf-8"))
    fine_to_ekman = {f: e for e, fines in ekman.items() for f in fines}             # 'annoyance' -> 'anger'

    def to_ekman(fine):
        targets = {fine_to_ekman[f] for f in fine if f in fine_to_ekman}            # 'neutral' has no mapping
        return targets.pop() if len(targets) == 1 else None                         # keep ONE clear emotion only

    out = {}
    for split, file in [("train", "train"), ("val", "dev"), ("test", "test")]:
        df = pd.read_csv(f"{data_dir}/{file}.tsv", sep="\t", header=None,
                         names=["text", "labels", "id"], quoting=3)
        df["fine"] = df["labels"].apply(lambda s: [names[int(i)] for i in str(s).split(",")])
        df["emotion"] = df["fine"].apply(to_ekman)
        out[split] = df
    return out


# ---------- 2) Text cleaning ----------
def clean_text(text):
    """Lowercase -> unify Reddit placeholders -> n't to 'not' -> keep letters only.
    Stop words are KEPT on purpose: 'not', 'so', 'never' carry emotion, and attention needs to see them."""
    text = str(text).lower().replace("\u2019", "'")          # curly apostrophe -> straight
    text = re.sub(r"\[name\]|\[religion\]", " name ", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


# ---------- 3) Vocabulary: word <-> integer ----------
class Vocab:
    """ID 0 = padding, ID 1 = <OOV> (a word never seen in training)."""

    def __init__(self, num_words=20000):
        self.num_words = num_words
        self.word_index = {}

    def fit(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        self.word_index = {"<OOV>": 1}
        for i, (w, _) in enumerate(counts.most_common(self.num_words - 2), start=2):
            self.word_index[w] = i
        return self

    def __len__(self):
        return len(self.word_index) + 1          # +1 for padding ID 0

    def encode(self, texts, max_len=MAX_LEN):
        """Cleaned texts -> (n, max_len) int array, POST padding (words first, zeros after)."""
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for r, t in enumerate(texts):
            ids = [self.word_index.get(w, 1) for w in t.split()][:max_len]
            out[r, :len(ids)] = ids
        return out

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_words": self.num_words, "word_index": self.word_index}, f)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        v = cls(d["num_words"]); v.word_index = d["word_index"]
        return v


# ---------- 4) GloVe embedding matrix ----------
def glove_matrix(glove_path, vocab, dim=100, seed=42):
    """Row i = GloVe vector of the word whose ID is i. Unknown words get small random vectors."""
    glove = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.asarray(parts[1:], dtype="float32")
    rng = np.random.default_rng(seed)
    mat = np.zeros((len(vocab), dim), dtype="float32")
    missing = []
    for w, i in vocab.word_index.items():
        if w in glove:
            mat[i] = glove[w]
        else:
            mat[i] = rng.normal(0, 0.6, dim); missing.append(w)
    return mat, glove, missing


# ---------- 5) Models ----------
class RNNClassifier(nn.Module):
    """Embedding (GloVe) -> LSTM or GRU -> last real hidden state -> 6 emotion scores."""

    def __init__(self, emb_matrix, rnn_type="LSTM", hidden=128, n_classes=6, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=False, padding_idx=0)
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(emb_matrix.shape[1], hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()                 # real words per comment
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lengths,
                                                   batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)                                      # packing = the RNN skips the padding
        h = h[0] if isinstance(h, tuple) else h                      # LSTM returns (h, c); GRU returns h
        return self.out(self.drop(h[-1])), None                      # (scores, no attention)


class BiLSTMAttention(nn.Module):
    """Embedding (GloVe) -> BiLSTM -> attention over ALL word states -> weighted sum -> 6 emotion scores."""

    def __init__(self, emb_matrix, hidden=128, n_classes=6, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=False, padding_idx=0)
        self.rnn = nn.LSTM(emb_matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.att_proj = nn.Linear(2 * hidden, 2 * hidden)            # W, b
        self.att_v = nn.Linear(2 * hidden, 1, bias=False)            # v
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        mask = x != 0                                                # True = real word, False = padding
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lengths,
                                                   batch_first=True, enforce_sorted=False)
        H, _ = self.rnn(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=x.size(1))  # (B, T, 2h)

        scores = self.att_v(torch.tanh(self.att_proj(H))).squeeze(-1)   # (B, T): one score per word
        scores = scores.masked_fill(~mask, -1e9)                     # padding can never get attention
        alpha = torch.softmax(scores, dim=1)                         # weights: positive, sum to 1
        context = (alpha.unsqueeze(-1) * H).sum(1)                   # weighted average of word states
        return self.out(self.drop(context)), alpha


def build_model(name, emb_matrix):
    if name == "BiLSTM+Attention":
        return BiLSTMAttention(emb_matrix)
    return RNNClassifier(emb_matrix, rnn_type=name)


# ---------- 6) Prediction helpers ----------
@torch.no_grad()
def predict_proba(model, X, device="cpu", batch=512):
    """Returns (probabilities (n, 6), attention (n, T) or None)."""
    model.eval()
    probs, atts = [], []
    for i in range(0, len(X), batch):
        logits, alpha = model(torch.as_tensor(X[i:i + batch], device=device))
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        if alpha is not None:
            atts.append(alpha.cpu().numpy())
    return np.concatenate(probs), (np.concatenate(atts) if atts else None)
