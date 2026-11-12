"""
app.py: Gradio web app for Emotion Classification.
Loads the best model chosen in the notebook (models/results.json), takes free text and returns
the dominant emotion + a confidence bar chart across all 6 emotions.
Run:  python app.py  -> open http://127.0.0.1:7860
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                     # draw charts without opening a window
import matplotlib.pyplot as plt
import gradio as gr

import emotion_utils as eu

with open("models/results.json") as f:
    info = json.load(f)
EMOTIONS = info["emotions"]

# use the best model whose weights are present (DistilBERT weights are too large for GitHub,
# so a fresh clone falls back to the best from-scratch model until the notebook is run)
ranked = sorted(info["results"], key=lambda m: info["results"][m]["f1_macro"], reverse=True)
BEST = next(m for m in ranked if os.path.exists(
    "models/DistilBERT/config.json" if m == "DistilBERT" else f"models/{m}/weights.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading best model:", BEST)

if BEST == "DistilBERT":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    hf_tok = AutoTokenizer.from_pretrained("models/DistilBERT")
    hf_model = AutoModelForSequenceClassification.from_pretrained("models/DistilBERT").to(DEVICE).eval()

    def get_probs(text):
        enc = hf_tok(text, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return torch.softmax(hf_model(**enc).logits, -1)[0].cpu().numpy()
else:
    vocab = eu.Vocab.load("models/vocab.json")
    model = eu.build_model(BEST, np.zeros((len(vocab), 100), dtype="float32"))   # shape only; weights loaded next
    model.load_state_dict(torch.load(f"models/{BEST}/weights.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    def get_probs(text):
        return eu.predict_proba(model, vocab.encode([eu.clean_text(text)]), DEVICE)[0][0]


COLORS = {"anger": "#d62728", "disgust": "#6b8e23", "fear": "#7f3fbf",
          "joy": "#f2b701", "sadness": "#1f77b4", "surprise": "#ff7f0e"}


def bar_chart(probs):
    order = np.argsort(probs)                                    # biggest bar on top
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([EMOTIONS[i] for i in order], probs[order],
            color=[COLORS[EMOTIONS[i]] for i in order])
    for y, i in enumerate(order):
        ax.text(probs[i] + 0.01, y, f"{probs[i]:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.12); ax.set_xlabel("confidence")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def classify(text):
    if not text or not text.strip():
        return "Please type some text.", None
    probs = get_probs(text)
    i = int(np.argmax(probs))
    return f"## {EMOTIONS[i].title()}\n**Confidence:** {probs[i]:.1%}", bar_chart(probs)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="Your text", placeholder="Type a sentence or comment..."),
    outputs=[gr.Markdown(label="Dominant emotion"), gr.Plot(label="Confidence across 6 emotions")],
    title="Emotion Classifier (GoEmotions, 6 emotions)",
    description=f"Detects joy, sadness, anger, fear, surprise or disgust. Model: **{BEST}**",
    examples=[
        ["This is the best day of my life, thank you all!"],
        ["I can't believe they cancelled the show, I'm devastated"],
        ["Stop lying to everyone, it's pathetic"],
        ["I'm so nervous about my exam tomorrow"],
        ["Wow, I had no idea that was even possible!"],
        ["That smell in the fridge is absolutely revolting"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
