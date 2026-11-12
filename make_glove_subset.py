"""
Builds data/glove_subset_100d.txt: only the GloVe 6B 100d vectors for words in the training vocabulary.
Keeps the repo small (15 MB instead of 347 MB).
Usage:  python make_glove_subset.py path/to/glove.6B.100d.txt
"""
import os
import sys
import emotion_utils as eu

glove_path = sys.argv[1]
train = eu.load_goemotions("data")["train"].dropna(subset=["emotion"])
vocab = set(eu.Vocab().fit(train["text"].apply(eu.clean_text)).word_index)

os.makedirs("data", exist_ok=True)
kept = 0
with open(glove_path, encoding="utf-8") as src, open("data/glove_subset_100d.txt", "w", encoding="utf-8") as out:
    for line in src:
        if line.split(" ", 1)[0] in vocab:
            out.write(line)
            kept += 1
print(f"kept {kept} of {len(vocab)} words")
