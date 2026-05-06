import numpy as np
import torch
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel, BertModel
from huggingface_hub import snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENA_MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-multi"
DNABERT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


# Load heavy encoder models once at module import time to avoid repeated initialization.
print("Loading GENA...")
gena_tokenizer = AutoTokenizer.from_pretrained(GENA_MODEL_NAME)
gena_model = AutoModel.from_pretrained(
    GENA_MODEL_NAME,
    output_hidden_states=True,
    trust_remote_code=True
).to(DEVICE)
gena_model.eval()

print("Loading DNABERT2...")
dnabert_tokenizer = AutoTokenizer.from_pretrained(
    DNABERT_MODEL_NAME,
    trust_remote_code=True
)

dnabert_model = BertModel.from_pretrained(
    DNABERT_MODEL_NAME,
    trust_remote_code=True
).to(DEVICE)
dnabert_model.eval()

def infer_types_from_sequences(seqs):
    """RNA if contains 'U', otherwise DNA."""
    types = []
    for s in seqs:
        if s is None:
            types.append("DNA")
        else:
            s = str(s).upper()
            types.append("RNA" if "U" in s else "DNA")
    return types


def onehot_with_type_bit(seqs, max_len=216):
    types = infer_types_from_sequences(seqs)
    alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    N = len(seqs)
    out = np.zeros((N, max_len * 4 + 1), dtype=np.float64)
    for i, (s, t) in enumerate(zip(seqs, types)):
        d = 1.0 if t == "RNA" else 0.0
        # Normalize RNA to DNA alphabet for a shared one-hot space.
        raw = (s or "").upper().replace("U", "T")
        raw = "".join(ch for ch in raw if ch in alphabet)[:max_len]
        for j, ch in enumerate(raw):
            out[i, j * 4 + alphabet[ch]] = 1.0
        out[i, -1] = d
    return out


def kmer_freq_with_type_bit(seqs, k=6):
    types = infer_types_from_sequences(seqs)
    vocab = [''.join(p) for p in product('ACGT', repeat=k)]
    vec = CountVectorizer(analyzer='char', ngram_range=(k, k),
                          lowercase=False, vocabulary=vocab)
    seqs_norm = [(s or "").upper().replace("U", "T") for s in seqs]
    X = vec.fit_transform(seqs_norm).astype(np.float64).toarray()
    # Convert raw k-mer counts into per-sequence frequencies.
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum
    d = np.array([1.0 if t == "RNA" else 0.0 for t in types], dtype=np.float64).reshape(-1, 1)
    return np.concatenate([X, d], axis=1)


@torch.no_grad()
def gena_embed(seqs, max_len=216, batch_size=64):
    types = infer_types_from_sequences(seqs)

    out = []
    for i in range(0, len(seqs), batch_size):

        batch = [(s or "").upper().replace("U", "T") for s in seqs[i:i + batch_size]]

        enc = gena_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)

        # Use the last hidden layer with attention-mask-aware mean pooling.
        h = gena_model(**enc).hidden_states[-1]

        m = enc["attention_mask"].unsqueeze(-1)

        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)

        out.append(pooled.cpu().numpy())

    E = np.vstack(out)

    d = np.array(
        [1.0 if t == "RNA" else 0.0 for t in types],
        dtype=np.float64
    ).reshape(-1, 1)

    return np.concatenate([E, d], axis=1)


@torch.no_grad()
def dnabert2_embed(seqs, pooling="mean", max_len=512, batch_size=32):

    types = infer_types_from_sequences(seqs)

    out = []

    for i in range(0, len(seqs), batch_size):

        batch = [str(s).upper().replace("U", "T") for s in seqs[i:i + batch_size]]

        inputs = dnabert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)

        hidden = dnabert_model(**inputs)[0]

        # Keep pooling configurable for quick ablations.
        if pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden.max(dim=1)[0]

        out.append(pooled.cpu().numpy())

    E = np.vstack(out)

    d = np.array(
        [1.0 if t == "RNA" else 0.0 for t in types],
        dtype=np.float64
    ).reshape(-1, 1)

    return np.concatenate([E, d], axis=1)