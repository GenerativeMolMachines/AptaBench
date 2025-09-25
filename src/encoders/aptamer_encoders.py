import numpy as np
import torch
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel



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
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum
    d = np.array([1.0 if t == "RNA" else 0.0 for t in types], dtype=np.float64).reshape(-1, 1)
    return np.concatenate([X, d], axis=1)


@torch.no_grad()
def gena_embed(seqs, model_name='AIRI-Institute/gena-lm-bert-base-t2t-multi',
               max_len=216, batch_size=64, device=None):
    types = infer_types_from_sequences(seqs)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,
                                      output_hidden_states=True,
                                      trust_remote_code=True).to(device)
    model.eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = [(s or "").upper().replace("U", "T") for s in seqs[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(pooled.detach().cpu().numpy())
    E = np.vstack(out)
    d = np.array([1.0 if t == "RNA" else 0.0 for t in types], dtype=np.float64).reshape(-1, 1)
    return np.concatenate([E, d], axis=1)


@torch.no_grad()
def dnabert2_embed(
    seqs,
    model_name="zhihan1996/DNABERT-2-117M",
    pooling="mean",
    max_len=512,
    batch_size=32,
    device=None
):
    """
    Generate embeddings for DNA/RNA sequences using DNABERT-2.

    Args:
        seqs (list[str]): List of sequences.
        model_name (str): Hugging Face model name.
        pooling (str): "mean" or "max" pooling.
        max_len (int): Maximum tokenized sequence length.
        batch_size (int): Number of sequences per batch.
        device (str): "cuda" or "cpu". If None, auto-detect.

    Returns:
        np.ndarray: Embeddings with an additional column encoding RNA/DNA type.
    """
    # infer types: RNA=1, DNA=0
    types = infer_types_from_sequences(seqs)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # model with safetensors
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True
    ).to(device)
    model.eval()

    out = []
    for i in range(0, len(seqs), batch_size):
        batch = [str(s).upper().replace("U", "T") for s in seqs[i:i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            hidden_states = model(**inputs)[0]

        if pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif pooling == "max":
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError("pooling must be 'mean' or 'max'")

        out.append(pooled.cpu().numpy())

    E = np.vstack(out)

    # add RNA/DNA flag
    d = np.array([1.0 if t == "RNA" else 0.0 for t in types], dtype=np.float64).reshape(-1, 1)

    return np.concatenate([E, d], axis=1)