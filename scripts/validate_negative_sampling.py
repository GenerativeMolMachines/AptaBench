from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.split import (
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
    stratified_group_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated negatives against shuffled baseline."
    )
    parser.add_argument("--dataset", default="dataset/AptaBench_dataset.csv")
    parser.add_argument("--outdir", default="artifacts/metrics/negative_sampling_validation")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument(
        "--split-mode",
        choices=("stratified", "disjoint_aptamer", "disjoint_molecule"),
        default="disjoint_molecule",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--fpr-target", type=float, default=0.05)
    return parser.parse_args()


def seq_features(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    n = max(len(seq), 1)
    return np.array(
        [
            len(seq),
            seq.count("A") / n,
            seq.count("C") / n,
            seq.count("G") / n,
            (seq.count("T") + seq.count("U")) / n,
            float("U" in seq),
        ],
        dtype=np.float32,
    )


def morgan_fp(smiles: str, n_bits: int = 512, radius: int = 2) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.uint8)
    mol = Chem.MolFromSmiles(str(smiles) if smiles else "")
    if mol is None:
        return arr
    fp = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df: pd.DataFrame) -> np.ndarray:
    seq_mat = np.vstack([seq_features(s) for s in df["sequence"].tolist()])
    mol_mat = np.vstack([morgan_fp(s) for s in df["canonical_smiles"].tolist()])
    return np.concatenate([seq_mat, mol_mat], axis=1).astype(np.float32)


def build_shuffled_negatives(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos_df = train_df[train_df["label"] == 1].copy()
    neg_n = int((train_df["label"] == 0).sum())

    known_pos = {
        (s, m) for s, m in zip(pos_df["sequence"].tolist(), pos_df["canonical_smiles"].tolist())
    }
    apt_pool = pos_df["sequence"].to_numpy()
    mol_pool = pos_df["canonical_smiles"].to_numpy()

    rows = []
    seen = set()
    attempts = 0
    max_attempts = max(neg_n * 30, 1000)
    while len(rows) < neg_n and attempts < max_attempts:
        attempts += 1
        pair = (
            apt_pool[int(rng.integers(0, len(apt_pool)))],
            mol_pool[int(rng.integers(0, len(mol_pool)))],
        )
        if pair in known_pos or pair in seen:
            continue
        seen.add(pair)
        rows.append(
            {
                "type": np.nan,
                "sequence": pair[0],
                "canonical_smiles": pair[1],
                "pKd_value": np.nan,
                "label": 0,
                "buffer": "shuffled_negative",
                "origin": "synthetic_shuffle",
                "source": "negative_sampling_validation",
            }
        )
    if len(rows) < neg_n:
        raise RuntimeError(
            f"Could not create enough shuffled negatives: {len(rows)} / {neg_n}. "
            "Try fewer seeds or a bigger train split."
        )
    return pd.DataFrame(rows)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    thresholds = np.unique(np.round(y_score, 8))
    best_recall = 0.0
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fpr = fp / max(fp + tn, 1)
        if fpr <= fpr_target:
            best_recall = max(best_recall, recall_score(y_true, y_pred, zero_division=0))
    return float(best_recall)


def evaluate_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, *, seed: int, n_estimators: int, fpr_target: float
) -> dict:
    X_train = build_features(train_df)
    y_train = train_df["label"].astype(int).to_numpy()
    X_test = build_features(test_df)
    y_test = test_df["label"].astype(int).to_numpy()

    # Keep the same LightGBM defaults used in notebook regression/classification runs.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    num_leaves=96,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=0.1,
                    random_state=seed,
                    n_jobs=2,
                    verbosity=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        y_score = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, y_score, n_bins=10, strategy="quantile")
    return {
        "pr_auc": average_precision_score(y_test, y_score),
        "roc_auc": roc_auc_score(y_test, y_score),
        "recall_at_fpr": recall_at_fpr(y_test, y_score, fpr_target=fpr_target),
        "brier": brier_score_loss(y_test, y_score),
        "calib_mae": float(np.mean(np.abs(frac_pos - mean_pred))),
        "scores": y_score,
    }


def matched_entity_check(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    def top_share(series: pd.Series) -> float:
        vc = series.value_counts(normalize=True)
        return float(vc.iloc[0]) if len(vc) else 0.0

    return {
        "aptamer_top_share_abs_diff": abs(top_share(df_a["sequence"]) - top_share(df_b["sequence"])),
        "molecule_top_share_abs_diff": abs(top_share(df_a["canonical_smiles"]) - top_share(df_b["canonical_smiles"])),
        "n_rows_equal": int(len(df_a) == len(df_b)),
    }


def negative_type_sanity(df_a: pd.DataFrame, df_b: pd.DataFrame, seed: int) -> float:
    dfa = df_a.copy()
    dfb = df_b.copy()
    dfa["neg_type"] = 0
    dfb["neg_type"] = 1
    df = pd.concat([dfa, dfb], ignore_index=True)

    X = build_features(df)
    y = df["neg_type"].astype(int).to_numpy()
    clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, p))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df = df.dropna(subset=["sequence", "canonical_smiles", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    seed_values = list(range(42, 42 + args.seeds))
    records: list[dict] = []
    sanity_records: list[dict] = []

    for seed in seed_values:
        if args.split_mode == "stratified":
            split = stratified_group_splits(df, n_splits=5, random_state=seed)[0]
        elif args.split_mode == "disjoint_molecule":
            split = disjoint_molecule_splits(df, n_splits=5, random_state=seed)[0]
        else:
            split = disjoint_aptamer_splits(df, n_splits=5, random_state=seed)[0]
        train_idx, test_idx = split
        train_real = df.iloc[train_idx].reset_index(drop=True)
        test_real = df.iloc[test_idx].reset_index(drop=True)

        train_a = train_real.copy()
        train_b_pos = train_real[train_real["label"] == 1].copy()
        train_b_neg = build_shuffled_negatives(train_real, seed=seed)
        train_b = pd.concat([train_b_pos, train_b_neg], ignore_index=True)

        res_a = evaluate_model(
            train_a, test_real, seed=seed, n_estimators=args.n_estimators, fpr_target=args.fpr_target
        )
        res_b = evaluate_model(
            train_b, test_real, seed=seed, n_estimators=args.n_estimators, fpr_target=args.fpr_target
        )

        for mode, res in [("curated", res_a), ("shuffled", res_b)]:
            records.append(
                {
                    "seed": seed,
                    "split_mode": args.split_mode,
                    "train_negative_mode": mode,
                    "pr_auc": res["pr_auc"],
                    "roc_auc": res["roc_auc"],
                    "recall_at_fpr": res["recall_at_fpr"],
                    "brier": res["brier"],
                    "calib_mae": res["calib_mae"],
                }
            )

        # Sanity checks for this seed.
        neg_a = train_a[train_a["label"] == 0]
        neg_b = train_b[train_b["label"] == 0]
        sanity_records.append(
            {
                "seed": seed,
                "negative_type_roc_auc": negative_type_sanity(neg_a, neg_b, seed=seed),
                "curated_easy_frac": float(np.mean(res_a["scores"] < 0.1)),
                "shuffled_easy_frac": float(np.mean(res_b["scores"] < 0.1)),
                **matched_entity_check(neg_a, neg_b),
            }
        )

    results_df = pd.DataFrame(records)
    sanity_df = pd.DataFrame(sanity_records)
    results_df.to_csv(outdir / "seed_level_results.csv", index=False)
    sanity_df.to_csv(outdir / "sanity_checks.csv", index=False)

    paired = []
    for metric in ["pr_auc", "roc_auc", "recall_at_fpr", "brier", "calib_mae"]:
        a = results_df[results_df["train_negative_mode"] == "curated"].sort_values("seed")[metric].to_numpy()
        b = results_df[results_df["train_negative_mode"] == "shuffled"].sort_values("seed")[metric].to_numpy()
        stat = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
        diff = a - b
        paired.append(
            {
                "metric": metric,
                "curated_mean": float(np.mean(a)),
                "shuffled_mean": float(np.mean(b)),
                "mean_diff_curated_minus_shuffled": float(np.mean(diff)),
                "p_value": float(stat.pvalue),
                "ci95_low": float(np.percentile(diff, 2.5)),
                "ci95_high": float(np.percentile(diff, 97.5)),
            }
        )
    summary_df = pd.DataFrame(paired)
    summary_df.to_csv(outdir / "summary_stats.csv", index=False)

    pr_pivot = results_df.pivot(index="seed", columns="train_negative_mode", values="pr_auc")
    plt.figure(figsize=(7, 4))
    plt.boxplot(
        [pr_pivot["curated"].values, pr_pivot["shuffled"].values],
        tick_labels=["Curated", "Shuffled"],
    )
    plt.ylabel("PR-AUC")
    plt.title(f"PR-AUC by negative strategy ({args.split_mode})")
    plt.tight_layout()
    plt.savefig(outdir / "pr_auc_boxplot.png", dpi=180)
    plt.close()

    winner = summary_df.loc[summary_df["metric"] == "pr_auc"].iloc[0]
    verdict = (
        "curated_better"
        if winner["mean_diff_curated_minus_shuffled"] > 0 and winner["p_value"] < 0.05
        else "inconclusive_or_not_better"
    )
    lines = [
        f"split_mode: {args.split_mode}",
        f"n_seeds: {args.seeds}",
        f"verdict: {verdict}",
        "",
        "Primary metric (PR-AUC):",
        f"- curated_mean: {winner['curated_mean']:.4f}",
        f"- shuffled_mean: {winner['shuffled_mean']:.4f}",
        f"- mean_diff: {winner['mean_diff_curated_minus_shuffled']:.4f}",
        f"- p_value: {winner['p_value']:.6f}",
        f"- ci95: [{winner['ci95_low']:.4f}, {winner['ci95_high']:.4f}]",
        "",
        "Sanity highlights:",
        f"- mean_negative_type_roc_auc: {sanity_df['negative_type_roc_auc'].mean():.4f}",
        f"- mean_curated_easy_frac: {sanity_df['curated_easy_frac'].mean():.4f}",
        f"- mean_shuffled_easy_frac: {sanity_df['shuffled_easy_frac'].mean():.4f}",
    ]
    (outdir / "conclusion.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
