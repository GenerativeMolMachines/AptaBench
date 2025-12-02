import os
import json
import hashlib
import gc
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna


def _compute_dataset_hash(df):
    """Compute hash of dataset for cache validation."""
    # Use shape, first few rows, and column names for hash
    data_str = f"{df.shape}_{df.iloc[:5].to_json()}_{list(df.columns)}"
    return hashlib.md5(data_str.encode()).hexdigest()


def _get_embedding_cache_dir(embeddings_dir="notebooks/data/embeddings"):
    """Get or create embeddings cache directory."""
    cache_dir = Path(embeddings_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_checkpoint_key(split_mode, apt_name, mol_name):
    """Generate unique key for checkpoint identification."""
    return (split_mode, apt_name, mol_name)


def precompute_and_save_embeddings(
    df,
    apt_cfgs,
    mol_cfgs,
    embeddings_dir="notebooks/data/embeddings",
    force_recompute=False,
):
    """
    Precompute and save embeddings for all encoders.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ["sequence", "canonical_smiles"].
    apt_cfgs : list of dict
        Aptamer encoder configs (name, func, kwargs).
    mol_cfgs : list of dict
        Molecule encoder configs (name, func, kwargs).
    embeddings_dir : str
        Directory to save embeddings.
    force_recompute : bool
        If True, recompute even if cached embeddings exist.
    
    Returns
    -------
    tuple (apt_map, mol_map, dataset_hash)
        Dictionary of aptamer embeddings, molecule embeddings, and dataset hash.
    """
    df = df.reset_index(drop=True)
    dataset_hash = _compute_dataset_hash(df)
    cache_dir = _get_embedding_cache_dir(embeddings_dir)
    
    # Paths for metadata
    metadata_file = cache_dir / f"metadata_{dataset_hash}.json"
    
    apt_map = {}
    mol_map = {}
    
    print(f"Dataset hash: {dataset_hash}")
    print(f"Cache directory: {cache_dir}")
    
    # Check if metadata exists and load if not force_recompute
    if metadata_file.exists() and not force_recompute:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load aptamer embeddings
        print("Loading cached aptamer embeddings...")
        for cfg in apt_cfgs:
            name = cfg["name"]
            cache_file = cache_dir / f"apt_{name}_{dataset_hash}.npy"
            if cache_file.exists():
                apt_map[name] = np.load(cache_file)
                print(f"  ✓ Loaded {name}: shape {apt_map[name].shape}")
            else:
                print(f"  ✗ Missing cache for {name}, recomputing...")
                apt_map[name] = cfg["func"](df["sequence"].tolist(), **cfg.get("kwargs", {}))
                np.save(cache_file, apt_map[name])
        
        # Load molecule embeddings
        print("Loading cached molecule embeddings...")
        for cfg in mol_cfgs:
            name = cfg["name"]
            cache_file = cache_dir / f"mol_{name}_{dataset_hash}.npy"
            if cache_file.exists():
                mol_map[name] = np.load(cache_file)
                print(f"  ✓ Loaded {name}: shape {mol_map[name].shape}")
            else:
                print(f"  ✗ Missing cache for {name}, recomputing...")
                mol_map[name] = cfg["func"](df["canonical_smiles"].tolist(), **cfg.get("kwargs", {}))
                np.save(cache_file, mol_map[name])
    else:
        # Compute all embeddings
        print("Computing aptamer embeddings...")
        for cfg in tqdm(apt_cfgs, desc="Aptamer encoders"):
            name = cfg["name"]
            print(f"  Computing {name}...")
            apt_map[name] = cfg["func"](df["sequence"].tolist(), **cfg.get("kwargs", {}))
            cache_file = cache_dir / f"apt_{name}_{dataset_hash}.npy"
            np.save(cache_file, apt_map[name])
            print(f"  ✓ Saved {name}: shape {apt_map[name].shape}")
        
        print("Computing molecule embeddings...")
        for cfg in tqdm(mol_cfgs, desc="Molecule encoders"):
            name = cfg["name"]
            print(f"  Computing {name}...")
            mol_map[name] = cfg["func"](df["canonical_smiles"].tolist(), **cfg.get("kwargs", {}))
            cache_file = cache_dir / f"mol_{name}_{dataset_hash}.npy"
            np.save(cache_file, mol_map[name])
            print(f"  ✓ Saved {name}: shape {mol_map[name].shape}")
        
        # Save metadata
        metadata = {
            "dataset_hash": dataset_hash,
            "dataset_shape": list(df.shape),
            "aptamer_encoders": [cfg["name"] for cfg in apt_cfgs],
            "molecule_encoders": [cfg["name"] for cfg in mol_cfgs],
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_file}")
    
    return apt_map, mol_map, dataset_hash


def load_embeddings(
    df,
    apt_cfgs,
    mol_cfgs,
    embeddings_dir="notebooks/data/embeddings",
):
    """
    Load precomputed embeddings from cache.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ["sequence", "canonical_smiles"].
    apt_cfgs : list of dict
        Aptamer encoder configs (name, func, kwargs).
    mol_cfgs : list of dict
        Molecule encoder configs (name, func, kwargs).
    embeddings_dir : str
        Directory where embeddings are saved.
    
    Returns
    -------
    tuple (apt_map, mol_map) or None
        Dictionary of aptamer and molecule embeddings, or None if cache miss.
    """
    df = df.reset_index(drop=True)
    dataset_hash = _compute_dataset_hash(df)
    cache_dir = _get_embedding_cache_dir(embeddings_dir)
    
    metadata_file = cache_dir / f"metadata_{dataset_hash}.json"
    
    if not metadata_file.exists():
        return None
    
    apt_map = {}
    mol_map = {}
    
    # Load aptamer embeddings
    all_apt_found = True
    for cfg in apt_cfgs:
        name = cfg["name"]
        cache_file = cache_dir / f"apt_{name}_{dataset_hash}.npy"
        if cache_file.exists():
            apt_map[name] = np.load(cache_file)
        else:
            all_apt_found = False
            break
    
    # Load molecule embeddings
    all_mol_found = True
    for cfg in mol_cfgs:
        name = cfg["name"]
        cache_file = cache_dir / f"mol_{name}_{dataset_hash}.npy"
        if cache_file.exists():
            mol_map[name] = np.load(cache_file)
        else:
            all_mol_found = False
            break
    
    if all_apt_found and all_mol_found:
        return apt_map, mol_map
    else:
        return None


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
    use_cached_embeddings=True,
    embeddings_dir="notebooks/data/embeddings",
    force_recompute_embeddings=False,
    checkpoint_dir="notebooks/checkpoints",
    results_file=None,
    resume=True,
    use_pruning=True
):
    """
    Run LightGBM screening with Optuna hyperparameter optimization,
    using precomputed JSON splits. Includes checkpointing and online saving.

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
    use_cached_embeddings : bool
        If True, try to load embeddings from cache first, compute if not found.
    embeddings_dir : str
        Directory for cached embeddings.
    force_recompute_embeddings : bool
        If True, recompute embeddings even if cached versions exist.
    checkpoint_dir : str
        Directory for storing checkpoints (Optuna DB files).
    results_file : str or None
        Path to CSV file for online saving of results. If None, uses default.
    resume : bool
        If True, skip already completed combinations and continue from checkpoint.
    use_pruning : bool
        If True, use MedianPruner for early stopping of bad trials.

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
    
    dataset_hash = _compute_dataset_hash(df)
    cache_dir = _get_embedding_cache_dir(embeddings_dir)
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if results_file is None:
        results_file = checkpoint_path / "results_incremental.csv"
    else:
        results_file = Path(results_file)
    
    completed_combinations = set()
    records = []
    if resume and results_file.exists():
        try:
            existing_df = pd.read_csv(results_file)
            records = existing_df.to_dict('records')
            for row in records:
                key = (row['split'], row['aptamer_encoder'], row['molecule_encoder'])
                completed_combinations.add(key)
            print(f"✓ Loaded {len(records)} existing results. Continuing with {len(completed_combinations)} combinations.")
        except Exception as e:
            print(f"⚠ Checkpoint loading error: {e}. Starting fresh.")
            records = []
            completed_combinations = set()

    #
    for split_mode in split_modes:
        splits = load_splits(split_mode, base_dir=splits_dir)

        for apt_cfg in tqdm(apt_cfgs, desc=f"Apt encoders ({split_mode})"):
            a_name = apt_cfg["name"]
            
            cache_file_apt = cache_dir / f"apt_{a_name}_{dataset_hash}.npy"
            if cache_file_apt.exists():
                Xa = np.load(cache_file_apt).astype(np.float32)
            else:
                if use_cached_embeddings:
                    print(f"⚠ Cache miss for {a_name}, skipping...")
                    continue
                Xa = apt_cfg["func"](df["sequence"].tolist(), **apt_cfg.get("kwargs", {})).astype(np.float32)
                np.save(cache_file_apt, Xa)
            
            for mol_cfg in mol_cfgs:
                m_name = mol_cfg["name"]
                
                cache_file_mol = cache_dir / f"mol_{m_name}_{dataset_hash}.npy"
                if cache_file_mol.exists():
                    Xm = np.load(cache_file_mol).astype(np.float32)
                else:
                    if use_cached_embeddings:
                        print(f"⚠ Cache miss for {m_name}, skipping...")
                        continue
                    Xm = mol_cfg["func"](df["canonical_smiles"].tolist(), **mol_cfg.get("kwargs", {})).astype(np.float32)
                    np.save(cache_file_mol, Xm)
                
                X = np.concatenate([Xa, Xm], axis=1).astype(np.float32)
                
                combo_key = (split_mode, a_name, m_name)
                if resume and combo_key in completed_combinations:
                    print(f"  ⏭ Skipping {a_name} × {m_name} (already done)")
                    continue

                study_name = f"{split_mode}_{a_name}_{m_name}_{dataset_hash}"
                study_db = checkpoint_path / f"{study_name}.db"
                
                def objective(trial):
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                        "max_depth": trial.suggest_int("max_depth", -1, 15),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                    }

                    scores = []
                    for fold_idx, (tr, va) in enumerate(splits):
                        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]

                        if scale:
                            scaler = StandardScaler()
                            Xtr = scaler.fit_transform(Xtr)
                            Xva = scaler.transform(Xva)

                        clf = LGBMClassifier(
                            **params,
                            random_state=random_state,
                            n_jobs=2,
                            verbosity=-1,
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                            if use_pruning:
                                clf.fit(
                                    Xtr, ytr,
                                    eval_set=[(Xva, yva)],
                                    callbacks=[optuna.integration.LightGBMPruningCallback(trial, "auc")],
                                    eval_metric="auc"
                                )
                            else:
                                clf.fit(Xtr, ytr)

                        s = clf.predict_proba(Xva)[:, 1]
                        yhat = clf.predict(Xva)

                        roc = roc_auc_score(yva, s) if len(np.unique(yva)) > 1 else np.nan
                        mcc = matthews_corrcoef(yva, yhat)

                        scores.append(roc if metric == "roc_auc" else mcc)
                        
                        if use_pruning and len(scores) >= 2:
                            trial.report(np.nanmean(scores), step=fold_idx)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                    
                    del clf, scaler, s, yhat, roc, mcc

                    return np.nanmean(scores)

                storage_url = f"sqlite:///{study_db}"
                
                pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1) if use_pruning else None
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    if n_completed > 0:
                        print(f"  ↻ Continuing study: {n_completed}/{n_trials} trials done")
                except:
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage_url,
                        direction="maximize",
                        pruner=pruner,
                        sampler=optuna.samplers.TPESampler(seed=random_state),
                        load_if_exists=True
                    )
                    n_completed = 0
                
                remaining_trials = max(0, n_trials - n_completed)
                if remaining_trials > 0:
                    study.optimize(
                        objective, 
                        n_trials=remaining_trials, 
                        show_progress_bar=False,
                        gc_after_trial=True
                    )
                
                best_params = study.best_params
                del study

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
                        n_jobs=2,
                        verbosity=-1,
                    )
                    clf.fit(Xtr, ytr)

                    s = clf.predict_proba(Xva)[:, 1]
                    yhat = clf.predict(Xva)

                    roc = roc_auc_score(yva, s) if len(np.unique(yva)) > 1 else np.nan
                    mcc = matthews_corrcoef(yva, yhat)

                    roc_scores.append(roc)
                    mcc_scores.append(mcc)

                new_record = {
                    "split": split_mode,
                    "aptamer_encoder": a_name,
                    "molecule_encoder": m_name,
                    "ROC-AUC mean": np.nanmean(roc_scores),
                    "ROC-AUC std": np.nanstd(roc_scores),
                    "MCC mean": np.nanmean(mcc_scores),
                    "MCC std": np.nanstd(mcc_scores),
                    "best_params": str(best_params),
                }
                records.append(new_record)
                
                try:
                    current_df = pd.DataFrame(records)
                    current_df.to_csv(results_file, index=False)
                    print(f"  ✓ Saved: {a_name} × {m_name} → {results_file}")
                except Exception as e:
                    print(f"  ⚠ Save error: {e}")
                
                try:
                    del clf, X, Xtr, Xva, scaler, best_params, roc_scores, mcc_scores
                    del s, yhat, roc, mcc
                except:
                    pass
                gc.collect()
            
            try:
                del Xm
            except:
                pass
            gc.collect()
        
        try:
            del Xa
        except:
            pass
        gc.collect()

    return pd.DataFrame.from_records(records)


def screen_lgbm_regression_simple(
    df,
    apt_map_full,
    mol_map_full,
    split_modes=("stratified", "disjoint_aptamer", "disjoint_molecule"),
    scale=True,
    random_state=42,
    splits_dir="dataset/splits",
):
    """
    Simple regression screening using preloaded embeddings.
    No hyperparameter optimization - uses sensible defaults.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset with pKd_value column.
    apt_map_full : dict
        Preloaded aptamer embeddings {name: array} for full dataset.
    mol_map_full : dict
        Preloaded molecule embeddings {name: array} for full dataset.
    
    Returns
    -------
    pandas.DataFrame
        Results with MAE, RMSE, R² for each encoder combination.
    """
    # Get indices of valid pKd samples
    valid_idx = df["pKd_value"].notna().values
    n_valid = valid_idx.sum()
    print(f"Regression dataset: {n_valid} samples with valid pKd (from {len(df)} total)")
    
    y = df["pKd_value"].to_numpy()  # Keep full size, filter inside folds
    
    # Use full embeddings (will filter inside folds)
    apt_map = apt_map_full
    mol_map = mol_map_full
    
    # Default LightGBM params (from best classification models)
    default_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 96,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 0.1,
        'random_state': random_state,
        'n_jobs': 2,
        'verbosity': -1,
    }
    
    records = []
    
    for split_mode in split_modes:
        print(f"\n{'='*60}")
        print(f"Processing: {split_mode}")
        print(f"{'='*60}")
        
        splits = load_splits(split_mode, base_dir=splits_dir)
        
        for a_name in tqdm(apt_map.keys(), desc=f"Aptamer encoders"):
            Xa = apt_map[a_name].astype(np.float32)
            
            for m_name in mol_map.keys():
                Xm = mol_map[m_name].astype(np.float32)
                
                # Concatenate
                X = np.concatenate([Xa, Xm], axis=1).astype(np.float32)
                
                # Evaluate on all folds
                mae_scores, rmse_scores, r2_scores = [], [], []
                
                for tr, va in splits:
                    # Filter to only samples with valid pKd
                    tr_valid_mask = valid_idx[tr]
                    va_valid_mask = valid_idx[va]
                    
                    tr_valid = tr[tr_valid_mask]
                    va_valid = va[va_valid_mask]
                    
                    Xtr, Xva = X[tr_valid], X[va_valid]
                    ytr, yva = y[tr_valid], y[va_valid]
                    
                    if scale:
                        scaler = StandardScaler()
                        Xtr = scaler.fit_transform(Xtr)
                        Xva = scaler.transform(Xva)
                    
                    # Train regressor
                    reg = LGBMRegressor(**default_params)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        reg.fit(Xtr, ytr)
                    
                    y_pred = reg.predict(Xva)
                    
                    mae_scores.append(mean_absolute_error(yva, y_pred))
                    rmse_scores.append(np.sqrt(mean_squared_error(yva, y_pred)))
                    r2_scores.append(r2_score(yva, y_pred))
                
                records.append({
                    "split": split_mode,
                    "aptamer_encoder": a_name,
                    "molecule_encoder": m_name,
                    "MAE mean": np.mean(mae_scores),
                    "MAE std": np.std(mae_scores),
                    "RMSE mean": np.mean(rmse_scores),
                    "RMSE std": np.std(rmse_scores),
                    "R2 mean": np.mean(r2_scores),
                    "R2 std": np.std(r2_scores),
                })
                
                # Clean up
                del X
                gc.collect()
    
    return pd.DataFrame.from_records(records)
