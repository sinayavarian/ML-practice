#!/usr/bin/env python3
"""
AutoFeatureSelector (vote-based feature selection)

Usage:
  python auto_feature_selector.py --data fifa19.csv --methods pearson chi-square rfe log-reg rf lgbm --k 30 --min-votes 1

Inputs:
  --data: path to a CSV dataset
  --methods: list of feature selection methods to run
            Allowed: pearson, chi-square, rfe, log-reg, rf, lgbm
  --k: number of best features to output (default: 30)
  --min-votes: minimum number of methods that must select a feature (default: 1)
  --no-lgbm: disable LightGBM even if installed

Output:
  Prints best features (top-k by vote count, then name), and prints a small preview table.
  You can also save the full vote table with --out-table path.csv

Notes:
  This script is tailored to the FIFA19-style columns used in the notebook:
    Numeric: Overall + football skill metrics
    Categorical: Preferred Foot, Position, Body Type, Nationality, Weak Foot
  Target:
    y = (Overall >= 87)
  If your dataset differs, edit preprocess_dataset() accordingly.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def cor_selector(X: pd.DataFrame, y: pd.Series, num_feats: int):
    """Pearson correlation (filter) - selects top |corr| features."""
    y_ = y.astype(int)
    cor_list = []
    for col in X.columns:
        cor = X[col].corr(y_)
        cor_list.append(0.0 if np.isnan(cor) else float(cor))

    cor_abs = np.abs(cor_list)
    idx = np.argsort(cor_abs)[-num_feats:]
    cor_feature = X.columns[idx].tolist()
    cor_support = np.array([c in cor_feature for c in X.columns], dtype=bool)
    return cor_support, cor_feature


def chi_squared_selector(X: pd.DataFrame, y: pd.Series, num_feats: int):
    """Chi-square (filter) - requires non-negative features."""
    X_ = X.copy()
    min_vals = X_.min()
    neg_cols = min_vals[min_vals < 0].index
    if len(neg_cols) > 0:
        X_[neg_cols] = X_[neg_cols].sub(min_vals[neg_cols], axis=1)

    y_ = y.astype(int)
    selector = SelectKBest(score_func=chi2, k=num_feats)
    selector.fit(X_, y_)
    chi_support = selector.get_support()
    chi_feature = X.columns[chi_support].tolist()
    return chi_support, chi_feature


def rfe_selector(X: pd.DataFrame, y: pd.Series, num_feats: int):
    """RFE (wrapper) using Logistic Regression estimator."""
    y_ = y.astype(int)
    estimator = LogisticRegression(max_iter=2000, solver="liblinear")
    selector = RFE(estimator=estimator, n_features_to_select=num_feats, step=1)
    selector.fit(X, y_)
    rfe_support = selector.get_support()
    rfe_feature = X.columns[rfe_support].tolist()
    return rfe_support, rfe_feature


def embedded_log_reg_selector(X: pd.DataFrame, y: pd.Series, num_feats: int, C: float = 0.1, threshold: float = 0.0):
    """Embedded (L1 Logistic Regression) using SelectFromModel."""
    y_ = y.astype(int)
    log_reg = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        max_iter=2000
    )
    selector = SelectFromModel(
        estimator=log_reg,
        max_features=num_feats,
        threshold=threshold
    )
    selector.fit(X, y_)
    support = selector.get_support()
    feats = X.columns[support].tolist()
    return support, feats


def embedded_rf_selector(X: pd.DataFrame, y: pd.Series, num_feats: int, n_estimators: int = 300):
    """Embedded (Random Forest) using SelectFromModel over feature_importances_."""
    y_ = y.astype(int)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    selector = SelectFromModel(
        estimator=rf,
        max_features=num_feats,
        threshold=-float("inf")
    )
    selector.fit(X, y_)
    support = selector.get_support()
    feats = X.columns[support].tolist()
    return support, feats


def embedded_lgbm_selector(X: pd.DataFrame, y: pd.Series, num_feats: int, n_estimators: int = 500, learning_rate: float = 0.05):
    """Embedded (LightGBM) using SelectFromModel over feature_importances_."""
    if not HAS_LGBM:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

    X_ = X.copy()
    # avoid whitespace warning
    X_.columns = X_.columns.str.replace(r"\s+", "_", regex=True)

    y_ = y.astype(int)
    pos = int((y_ == 1).sum())
    neg = int((y_ == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1
    )
    selector = SelectFromModel(
        estimator=lgbm,
        max_features=num_feats,
        threshold=-float("inf")
    )
    selector.fit(X_, y_)
    support = selector.get_support()
    feats = X_.columns[support].tolist()
    return support, feats


def preprocess_dataset(dataset_path: str):
    """FIFA-style preprocessing: select columns, one-hot encode, build X and y."""
    df = pd.read_csv(dataset_path)

    numcols = [
        'Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling',
        'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
        'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower',
        'Strength', 'LongShots', 'Aggression', 'Interceptions'
    ]
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    missing = [c for c in (numcols + catcols) if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset missing required columns for this script. Missing: "
            + ", ".join(missing)
            + "\nEdit preprocess_dataset() to match your dataset."
        )

    df = df[numcols + catcols]

    traindf = pd.concat([df[numcols], pd.get_dummies(df[catcols])], axis=1)
    traindf = traindf.dropna()

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    num_feats = 30
    return X, y, num_feats


def autoFeatureSelector(dataset_path: str, methods: list[str] | None = None, k: int | None = None, min_votes: int = 1, use_lgbm: bool = True):
    if methods is None:
        methods = ['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm']

    X, y, default_k = preprocess_dataset(dataset_path)
    if k is None:
        k = default_k

    # scaled version for linear models
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    results: dict[str, np.ndarray] = {}

    if 'pearson' in methods:
        support, _ = cor_selector(X, y, k)
        results['Pearson'] = support

    if 'chi-square' in methods:
        support, _ = chi_squared_selector(X, y, k)
        results['Chi-2'] = support

    if 'rfe' in methods:
        support, _ = rfe_selector(X_scaled, y, k)
        results['RFE'] = support

    if 'log-reg' in methods:
        support, _ = embedded_log_reg_selector(X_scaled, y, k, C=0.1, threshold=0.0)
        results['LogReg(L1)'] = support

    if 'rf' in methods:
        support, _ = embedded_rf_selector(X, y, k)
        results['Random Forest'] = support

    if 'lgbm' in methods and use_lgbm:
        support, _ = embedded_lgbm_selector(X, y, k)
        results['LightGBM'] = support

    if not results:
        raise ValueError("No valid methods selected. Choose from: pearson, chi-square, rfe, log-reg, rf, lgbm")

    vote_df = pd.DataFrame({'Feature': list(X.columns)})
    for name, support in results.items():
        vote_df[name] = support.astype(bool)

    vote_cols = list(results.keys())
    vote_df['Total'] = vote_df[vote_cols].sum(axis=1)

    vote_df = vote_df.sort_values(['Total', 'Feature'], ascending=False).reset_index(drop=True)

    filtered = vote_df[vote_df['Total'] >= int(min_votes)]
    best_features = filtered.head(int(k))['Feature'].tolist()

    return best_features, vote_df


def parse_args(argv: list[str]):
    p = argparse.ArgumentParser(description="Vote-based AutoFeatureSelector (FIFA-style preprocessing).")

    p.add_argument("--data", required=True, help="Path to CSV dataset.")
    p.add_argument("--methods", nargs="+", default=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'],
                   help="Methods: pearson chi-square rfe log-reg rf lgbm")
    p.add_argument("--k", type=int, default=30, help="Number of best features to output (top-k by votes).")
    p.add_argument("--min-votes", type=int, default=1, help="Minimum votes required for a feature to be included.")
    p.add_argument("--out-table", default=None, help="Optional path to save the full vote table as CSV.")
    p.add_argument("--no-lgbm", action="store_true", help="Disable LightGBM even if installed.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    use_lgbm = (not args.no_lgbm)

    if 'lgbm' in args.methods and use_lgbm and not HAS_LGBM:
        print("ERROR: 'lgbm' requested but LightGBM is not installed. Install with: pip install lightgbm", file=sys.stderr)
        return 2

    best_features, vote_df = autoFeatureSelector(
        dataset_path=args.data,
        methods=args.methods,
        k=args.k,
        min_votes=args.min_votes,
        use_lgbm=use_lgbm
    )

    print(f"\nSelected methods: {args.methods}")
    print(f"Top features returned: {len(best_features)}\n")
    for i, f in enumerate(best_features, start=1):
        print(f"{i:>2}. {f}")

    print("\nVote table preview (top 20):")
    print(vote_df.head(20).to_string(index=False))

    if args.out_table:
        vote_df.to_csv(args.out_table, index=False)
        print(f"\nSaved full vote table to: {args.out_table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
