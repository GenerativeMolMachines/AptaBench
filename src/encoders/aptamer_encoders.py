import numpy as np
import torch
from contextlib import nullcontext
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel, BertModel
from huggingface_hub import snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENA_MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-multi"
DNABERT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
RINALMO_MODEL_NAME = "mega-v1"
RNAFM_REPR_LAYER = 12

gena_tokenizer = None
gena_model = None
dnabert_tokenizer = None
dnabert_model = None
rnafm_model = None
rnafm_alphabet = None
rnafm_batch_converter = None
rinalmo_model = None
rinalmo_alphabet = None


def _load_gena():
    """Load GENA only when its encoder is requested."""
    global gena_tokenizer, gena_model
    if gena_model is None:
        print("Loading GENA...")
        gena_tokenizer = AutoTokenizer.from_pretrained(GENA_MODEL_NAME)
        gena_model = AutoModel.from_pretrained(
            GENA_MODEL_NAME,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(DEVICE)
        gena_model.eval()
    return gena_tokenizer, gena_model


def _disable_dnabert_flash_attn() -> None:
    """DNABERT-2 optional Triton flash-attn is unavailable on Windows.

    ``flash_attn_triton.py`` imports ``triton`` at top level, so transformers'
    import checker and module exec both fail hard. Replace cached copies with
    an ImportError stub so ``bert_layers.py`` falls back to PyTorch attention.
    """
    import os
    from pathlib import Path

    stub = (
        'raise ImportError(\n'
        '    "DNABERT-2 Triton flash-attn disabled; using PyTorch attention"\n'
        ')\n'
    )
    roots = []
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    roots.append(hf_home / "modules" / "transformers_modules" / "zhihan1996")
    roots.append(hf_home / "hub" / "models--zhihan1996--DNABERT-2-117M")
    roots.append(Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "zhihan1996")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "models--zhihan1996--DNABERT-2-117M")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("flash_attn_triton.py"):
            try:
                if path.is_symlink() or path.stat().st_size != len(stub.encode("utf-8")):
                    # Prefer writing beside the symlink target when possible.
                    target = path.resolve() if path.is_symlink() else path
                    target.write_text(stub, encoding="utf-8")
                elif path.read_text(encoding="utf-8") != stub:
                    path.write_text(stub, encoding="utf-8")
            except OSError:
                # Symlink may be read-only; write a real file over the link.
                path.unlink(missing_ok=True)
                path.write_text(stub, encoding="utf-8")


def _load_dnabert2():
    """Load DNABERT2 only when its encoder is requested."""
    global dnabert_tokenizer, dnabert_model
    if dnabert_model is None:
        print("Loading DNABERT2...")
        _disable_dnabert_flash_attn()
        # transformers>=4.55 refuses torch.load on torch<2.6 even for local
        # pytorch_model.bin; DNABERT-2-117M ships .bin only. Allow local load.
        try:
            from transformers.utils import import_utils as _xf_import_utils
            import transformers.modeling_utils as _xf_modeling_utils

            _xf_import_utils.check_torch_load_is_safe = lambda: None
            _xf_modeling_utils.check_torch_load_is_safe = lambda: None
        except Exception:
            pass
        dnabert_tokenizer = AutoTokenizer.from_pretrained(
            DNABERT_MODEL_NAME,
            trust_remote_code=True,
        )
        dnabert_model = AutoModel.from_pretrained(
            DNABERT_MODEL_NAME,
            trust_remote_code=True,
        ).to(DEVICE)
        # The bundled Triton kernel targets an older Triton API. A nonzero
        # branch flag selects DNABERT2's equivalent PyTorch attention path;
        # the actual Dropout modules remain disabled in eval mode.
        for module in dnabert_model.modules():
            if module.__class__.__name__ == "BertUnpadSelfAttention":
                module.p_dropout = np.finfo(np.float32).eps
        dnabert_model.eval()
    return dnabert_tokenizer, dnabert_model


def _ensure_rnafm_weights():
    """Fetch official RNA-FM .pth weights (CUHK mirror via Hugging Face)."""
    from pathlib import Path
    import shutil

    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "RNA-FM_pretrained.pth"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Downloading RNA-FM weights requires huggingface-hub. "
            "Install with: pip install huggingface-hub"
        ) from exc
    print("Downloading official RNA-FM weights from Hugging Face (cuhkaih/rnafm)...")
    downloaded = Path(
        hf_hub_download(
            repo_id="cuhkaih/rnafm",
            filename="RNA-FM_pretrained.pth",
        )
    )
    if downloaded.resolve() != destination.resolve():
        shutil.copy2(downloaded, destination)
    return destination


def _load_rnafm():
    """Load official ml4bio RNA-FM via the `fm` / `rna-fm` package."""
    global rnafm_model, rnafm_alphabet, rnafm_batch_converter
    if rnafm_model is None:
        try:
            import fm
        except ImportError as exc:
            raise ImportError(
                "RNA-FM requires the official `rna-fm` package "
                "(import name `fm`). Install with: pip install rna-fm"
            ) from exc
        print("Loading official RNA-FM (fm.pretrained.rna_fm_t12)...")
        weights = _ensure_rnafm_weights()
        # Prefer local weights: the CUHK download endpoint often returns HTTP 403.
        rnafm_model, rnafm_alphabet = fm.pretrained.rna_fm_t12(str(weights))
        rnafm_model = rnafm_model.to(DEVICE)
        rnafm_model.eval()
        rnafm_batch_converter = rnafm_alphabet.get_batch_converter()
    return rnafm_model, rnafm_batch_converter


def _load_rinalmo():
    """Load official lbcb-sci RiNALMo-Mega with Zenodo weights (no flash-attn)."""
    global rinalmo_model, rinalmo_alphabet
    if rinalmo_model is None:
        try:
            import torch
            from rinalmo.config import model_config
            from rinalmo.data.alphabet import Alphabet
            from rinalmo.model.model import RiNALMo
        except ImportError as exc:
            raise ImportError(
                "RiNALMo requires the official lbcb-sci/RiNALMo package and its "
                "dependencies (einops, ml_collections, gdown). "
                "Install with: git clone https://github.com/lbcb-sci/RiNALMo "
                "&& cd RiNALMo && pip install . && pip install einops ml_collections gdown"
            ) from exc

        print(f"Loading official RiNALMo ({RINALMO_MODEL_NAME})...")
        weights_path = _ensure_rinalmo_mega_weights()
        config = model_config("mega")
        # Keep FlashAttention module layout so Zenodo weights load; the local
        # RiNALMo patch falls back to SDPA when flash-attn is unavailable.
        config.model.transformer.use_flash_attn = True
        model = RiNALMo(config)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        rinalmo_model = model.to(device=DEVICE)
        rinalmo_model.eval()
        rinalmo_alphabet = Alphabet(**config["alphabet"])
    return rinalmo_model, rinalmo_alphabet


def _ensure_rinalmo_mega_weights():
    """Download RiNALMo-Mega pretrained weights from Zenodo if missing."""
    from pathlib import Path
    import urllib.request

    cache_dir = Path.home() / ".cache" / "rinalmo_pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "rinalmo_mega_pretrained.pt"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    url = "https://zenodo.org/records/15043668/files/rinalmo_mega_pretrained.pt"
    print(f"Downloading RiNALMo-Mega weights to {destination}...")
    tmp = destination.with_suffix(".pt.partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)
    return destination

def release_rna_native_models():
    """Release RNA-native singleton weights between large embedding jobs."""
    global rnafm_model, rnafm_alphabet, rnafm_batch_converter
    global rinalmo_model, rinalmo_alphabet
    for model in (rnafm_model, rinalmo_model):
        if model is not None:
            model.to("cpu")
    rnafm_model = None
    rnafm_alphabet = None
    rnafm_batch_converter = None
    rinalmo_model = None
    rinalmo_alphabet = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _prepare_rna_sequences(seqs):
    """Validate RNA input while preserving uracil for RNA-native tokenizers."""
    prepared = []
    for index, sequence in enumerate(seqs):
        value = "" if sequence is None else str(sequence).upper()
        if not value:
            raise ValueError(f"RNA sequence at index {index} is empty")
        invalid = set(value) - set("ACGUN")
        if invalid:
            raise ValueError(
                f"RNA sequence at index {index} contains unsupported symbols: "
                f"{sorted(invalid)}"
            )
        prepared.append(value)
    return prepared


def _mean_pool_token_rows(hidden, lengths):
    """Mean-pool contiguous nucleotide token rows for each sequence."""
    pooled = []
    for row, length in zip(hidden, lengths):
        length = int(length)
        if length < 1:
            raise ValueError("RNA sequence produced no nucleotide tokens")
        # ESM/RNA-FM style: CLS/BOS at 0, nucleotides at 1:1+L, EOS after.
        pooled.append(row[1 : 1 + length].mean(dim=0))
    return torch.stack(pooled, dim=0)


@torch.no_grad()
def rnafm_embed(seqs, max_len=512, batch_size=16):
    """Official RNA-FM embeddings with U preserved (no U→T rewrite)."""
    prepared = _prepare_rna_sequences(seqs)
    if not prepared:
        return np.empty((0, 0), dtype=np.float32)
    model, batch_converter = _load_rnafm()
    output = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        truncated = [seq[:max_len] for seq in batch]
        data = [(f"seq_{start + i}", seq) for i, seq in enumerate(truncated)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        results = model(tokens, repr_layers=[RNAFM_REPR_LAYER])
        hidden = results["representations"][RNAFM_REPR_LAYER]
        lengths = [len(seq) for seq in truncated]
        pooled = _mean_pool_token_rows(hidden, lengths)
        output.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(output)


@torch.no_grad()
def rinalmo_embed(seqs, max_len=1022, batch_size=16):
    """Official RiNALMo-Mega embeddings with U preserved (no U→T rewrite)."""
    prepared = _prepare_rna_sequences(seqs)
    if not prepared:
        return np.empty((0, 0), dtype=np.float32)
    model, alphabet = _load_rinalmo()
    output = []
    total = len(prepared)
    for start in range(0, total, batch_size):
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"  RiNALMo embedding {min(start + batch_size, total)}/{total}")
        batch = [seq[:max_len] for seq in prepared[start : start + batch_size]]
        tokens = torch.tensor(
            alphabet.batch_tokenize(batch),
            dtype=torch.int64,
            device=DEVICE,
        )
        # Keep FP32: official AMP + torch.jit RoPE can fail NVRTC on some CUDA setups.
        outputs = model(tokens)
        hidden = outputs["representation"]
        lengths = [len(seq) for seq in batch]
        if hidden.ndim == 2:
            pooled = hidden
        else:
            pooled = _mean_pool_token_rows(hidden, lengths)
        output.append(
            pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)
        )
    return np.vstack(output)


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
    tokenizer, model = _load_gena()
    types = infer_types_from_sequences(seqs)

    out = []
    for i in range(0, len(seqs), batch_size):

        batch = [(s or "").upper().replace("U", "T") for s in seqs[i:i + batch_size]]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)

        # Use the last hidden layer with attention-mask-aware mean pooling.
        h = model(**enc).hidden_states[-1]

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
    tokenizer, model = _load_dnabert2()
    types = infer_types_from_sequences(seqs)

    out = []

    for i in range(0, len(seqs), batch_size):

        batch = [str(s).upper().replace("U", "T") for s in seqs[i:i + batch_size]]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)

        hidden = model(**inputs)[0]

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
