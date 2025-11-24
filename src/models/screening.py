import os
import json
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import optuna


def load_splits(split_mode, base_dir="..\dataset\splits"):
    """
    Load precomputed splits from JSON file.

    Parameters
    ----------
    split_mode : str
        One of {"stratified", "disjoint_aptamer", "disjoint_molecule"}.
    base_dir : str
        Path to the directory containing split JSONs.

    Returns
    -------
    list of (train_idx, val_idx)
        Indices for each fold.
    """
    path = os.path.join(base_dir, f"{split_mode}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return [(np.array(d["train_idx"]), np.array(d["val_idx"])) for d in data]


def screen_lgbm_optuna(
    df,
    apt_cfgs,
    mol_cfgs,
    split_modes=("stratified", "disjoint_aptamer", "disjoint_molecule"),
    scale=True,
    n_trials=20,
    metric="roc_auc",
    random_state=42,
    splits_dir="dataset/splits",
):
    """
    Run LightGBM screening with Optuna hyperparameter optimization,
    using precomputed JSON splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ["sequence", "canonical_smiles", "label"].
    apt_cfgs : list of dict
        Aptamer encoder configs (name, func, kwargs).
    mol_cfgs : list of dict
        Molecule encoder configs (name, func, kwargs).
    split_modes : tuple of str
        {"stratified", "disjoint_aptamer", "disjoint_molecule"}.
    scale : bool
        Apply StandardScaler.
    n_trials : int
        Number of Optuna trials per (apt × mol × split).
    metric : str
        Optimization metric: "roc_auc" or "mcc".
    random_state : int
        Random seed.
    splits_dir : str
        Directory with precomputed JSON splits.

    Returns
    -------
    pandas.DataFrame
        Results with columns:
        ["split", "aptamer_encoder", "molecule_encoder",
         "ROC-AUC mean", "ROC-AUC std", "MCC mean", "MCC std",
         "best_params"...]
    """
    df = df.reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()

    # --- precompute features ---
    apt_map = {
        cfg["name"]: cfg["func"](df["sequence"].tolist(), **cfg.get("kwargs", {}))
        for cfg in apt_cfgs
    }
    mol_map = {
        cfg["name"]: cfg["func"](df["canonical_smiles"].tolist(), **cfg.get("kwargs", {}))
        for cfg in mol_cfgs
    }

    records = []

    # --- loop over splits / encoders ---
    for split_mode in split_modes:
        splits = load_splits(split_mode, base_dir=splits_dir)

        for a_name, Xa in tqdm(apt_map.items(), desc=f"Apt encoders ({split_mode})"):
            for m_name, Xm in mol_map.items():
                X = np.concatenate([Xa, Xm], axis=1)

                def objective(trial):
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                        "max_depth": trial.suggest_int("max_depth", -1, 20),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    }

                    scores = []
                    for tr, va in splits:
                        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]

                        if scale:
                            scaler = StandardScaler()
                            Xtr = scaler.fit_transform(Xtr)
                            Xva = scaler.transform(Xva)

                        clf = LGBMClassifier(
                            **params,
                            random_state=random_state,
                            n_jobs=-1,
                            verbosity=-1,
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                            clf.fit(Xtr, ytr)

                        s = clf.predict_proba(Xva)[:, 1]
                        yhat = clf.predict(Xva)

                        roc = roc_auc_score(yva, s) if len(np.unique(yva)) > 1 else np.nan
                        mcc = matthews_corrcoef(yva, yhat)

                        scores.append(roc if metric == "roc_auc" else mcc)

                    return np.nanmean(scores)

                # --- run Optuna ---
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                best_params = study.best_params

                # --- evaluate best params ---
                roc_scores, mcc_scores = [], []
                for tr, va in splits:
                    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]

                    if scale:
                        scaler = StandardScaler()
                        Xtr = scaler.fit_transform(Xtr)
                        Xva = scaler.transform(Xva)

                    clf = LGBMClassifier(
                        **best_params,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    )
                    clf.fit(Xtr, ytr)

                    s = clf.predict_proba(Xva)[:, 1]
                    yhat = clf.predict(Xva)

                    roc = roc_auc_score(yva, s) if len(np.unique(yva)) > 1 else np.nan
                    mcc = matthews_corrcoef(yva, yhat)

                    roc_scores.append(roc)
                    mcc_scores.append(mcc)

                records.append(
                    {
                        "split": split_mode,
                        "aptamer_encoder": a_name,
                        "molecule_encoder": m_name,
                        "ROC-AUC mean": np.nanmean(roc_scores),
                        "ROC-AUC std": np.nanstd(roc_scores),
                        "MCC mean": np.nanmean(mcc_scores),
                        "MCC std": np.nanstd(mcc_scores),
                        "best_params": best_params,
                    }
                )

    return pd.DataFrame.from_records(records)
