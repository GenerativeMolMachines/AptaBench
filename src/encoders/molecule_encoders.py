import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator as rfg, MACCSkeys
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from transformers import AutoTokenizer, AutoModel
from unimol_tools import UniMolRepr


def morgan_fp(smiles_list, n_bits=2048, radius=2, counts=False):
    X = np.zeros((len(smiles_list), n_bits), dtype=np.int32 if counts else np.uint8)
    gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi) if smi else "")
        if mol is None:
            continue
        if counts:
            fp = gen.GetCountFingerprint(mol)
            for idx, val in fp.GetNonzeroElements().items():
                if idx < n_bits:
                    X[i, idx] = val
        else:
            fp = gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def maccs_fp(smiles_list):
    X = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi) if smi else "")
        if mol is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i] = arr
    return X


def physchem_descriptors(smiles_list):
    names = [n for n, _ in Descriptors._descList]
    calc = MolecularDescriptorCalculator(names)
    N, D = len(smiles_list), len(names)
    X = np.full((N, D), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi) if smi else "")
        if mol is None:
            continue
        try:
            vals = np.asarray(calc.CalcDescriptors(mol), dtype=np.float64)
            vals[~np.isfinite(vals)] = np.nan
            X[i] = vals
        except Exception:
            pass
    keep = ~np.isnan(X).any(axis=0)
    X = X[:, keep]
    X = np.clip(X, -1e9, 1e9).astype(np.float32, copy=False)
    return X


@torch.no_grad()
def chemberta_embed(smiles_list, model_name="seyonec/ChemBERTa-zinc-base-v1",
                    batch_size=64, max_len=128, pooling="mean", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device).eval()
    out = []
    for i in range(0, len(smiles_list), batch_size):
        batch = [str(s) if s is not None else "" for s in smiles_list[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        h = mdl(**enc).last_hidden_state
        if pooling == "cls":
            pooled = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(pooled.detach().cpu().numpy())
    return np.vstack(out)


def unimol_embed(smiles_list, model_name="unimolv1", model_size="84m", return_atomic=True):
    clf = UniMolRepr(data_type='molecule', remove_hs=False,
                     model_name=model_name, model_size=model_size)
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=return_atomic)
    if return_atomic:
        return np.array(unimol_repr['cls_repr'])
    return np.array(unimol_repr['cls_repr'])


@torch.no_grad()
def molformer_embed(smiles_list, model_name="ibm/MoLFormer-XL-both-10pct",
                    max_len=512, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name,
                                      deterministic_eval=True,
                                      trust_remote_code=True).to(device)
    model.eval()
    out = []
    for i in range(0, len(smiles_list), batch_size):
        batch = [str(s) if s is not None else "" for s in smiles_list[i:i + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        outputs = model(**inputs)
        pooled = outputs.pooler_output.detach().cpu().numpy()
        out.append(pooled)
    return np.vstack(out)
