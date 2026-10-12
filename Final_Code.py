#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROAD NOISE RISK LEVEL PREDICTION USING MACHINE LEARNING
Complete Comparative Analysis of Four Datasets
==============================================
ONE standalone script -> FOUR independent dataset analyses -> ONE PDF report.

For EACH dataset (never merged, never mixed):
  cleaning -> 80/20, 70/30, 60/40 splits -> preprocessing fitted on TRAIN only
  (median imputation, IQR outlier clipping, standard scaling) -> LR, LMR, XGB, RF
  -> Accuracy / weighted Precision / Recall / F1 (+ macro) -> graphs
  -> robustness testing (Gaussian noise, 20/30/40 % of the training std, models fixed)
  -> confusion matrices (80/20) -> consistency analysis -> best model -> CSV files.
Afterwards a cross-dataset COMPARISON section is built from the stored results only.

Usage:
    python road_noise_four_datasets_analysis.py                (auto-detect datasets)
    python road_noise_four_datasets_analysis.py path/to/folder (folder holding the datasets)

Model interpretation
--------------------
LR  = Logistic Regression (multinomial classifier)
LMR = Linear / Multi-linear Regression BASELINE (not a native classifier): trained on the
      ordinal class codes (Low < Moderate < High), it predicts a continuous value which is
      rounded and clipped to the nearest valid class code before classification metrics.
XGB = XGBoost classifier
RF  = Random Forest classifier

Required packages:
    pip install numpy pandas scikit-learn matplotlib openpyxl xgboost reportlab
"""

import os
import sys
import re
import glob
import warnings
import importlib

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Dependency check
# --------------------------------------------------------------------------
_REQUIRED = [
    ("numpy", "numpy", "pip install numpy"),
    ("pandas", "pandas", "pip install pandas"),
    ("sklearn", "scikit-learn", "pip install scikit-learn"),
    ("matplotlib", "matplotlib", "pip install matplotlib"),
    ("openpyxl", "openpyxl", "pip install openpyxl"),
    ("xgboost", "xgboost", "pip install xgboost"),
    ("reportlab", "reportlab", "pip install reportlab"),
]
_missing = []
for _mod, _pkg, _cmd in _REQUIRED:
    try:
        importlib.import_module(_mod)
    except ImportError:
        _missing.append((_pkg, _cmd))
if _missing:
    print("=" * 70)
    print("ERROR: missing required Python package(s). Install with:")
    for _pkg, _cmd in _missing:
        print(f"   {_cmd}")
    print("Or install everything at once:")
    print("   pip install numpy pandas scikit-learn matplotlib openpyxl xgboost reportlab")
    print("=" * 70)
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from xgboost import XGBClassifier

from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.utils import ImageReader

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
RESULTS_DIR = "results"
PDF_PATH = os.path.join(RESULTS_DIR, "Road_Noise_Four_Datasets_Complete_Analysis.pdf")
N_DATASETS = 4
FIG_DPI = 150

REQUIRED_FEATURES = ["AVG_NOISE_LVL", "MEU", "Duration_hours", "Cumulative_MEU"]
TARGET = "Risk_Level"
REQUIRED_COLUMNS = REQUIRED_FEATURES + [TARGET]

MODELS = ["LR", "LMR", "XGB", "RF"]
SPLITS = [(0.20, "80/20"), (0.30, "70/30"), (0.40, "60/40")]
SPLIT_LABELS = [l for _, l in SPLITS]
PRIMARY_SPLIT = "80/20"                 # primary split for robustness plots and confusion matrices
PERTURBATIONS = [20, 30, 40]            # robustness noise levels (%), NOT split sizes
OUTLIER_IQR_FACTOR = 3.0                # clip beyond Q1 - 3*IQR / Q3 + 3*IQR (no rows deleted)

MODEL_COLORS = {"LR": "#1f77b4", "LMR": "#ff7f0e", "XGB": "#2ca02c", "RF": "#d62728"}
SPLIT_COLORS = {"80/20": "#4c72b0", "70/30": "#dd8452", "60/40": "#55a868"}
SPLIT_STYLES = {"80/20": "-", "70/30": "--", "60/40": ":"}
DATASET_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def split_tag(label):
    return label.replace("/", "_")


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def short_name(name, n=24):
    return name if len(name) <= n else name[:n - 3] + "..."


def base_model(s):
    """'XGB (tied with RF)' -> 'XGB'"""
    return str(s).split(" (")[0]


# --------------------------------------------------------------------------
# 1. Dataset detection (every sheet of every file is inspected)
# --------------------------------------------------------------------------
def norm_col(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


_NORM_TO_CANON = {norm_col(c): c for c in REQUIRED_COLUMNS}


def match_columns(columns):
    mapping = {}
    for c in columns:
        n = norm_col(c)
        if n in _NORM_TO_CANON and _NORM_TO_CANON[n] not in mapping.values():
            mapping[c] = _NORM_TO_CANON[n]
    return mapping


def read_csv_any(path):
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:  # noqa
            last = e
    raise last


def find_datasets(arg=None):
    """Return one candidate (best sheet) per dataset file, sorted by file name."""
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    search_dirs, files = [], []
    if arg and os.path.isfile(arg):
        files = [arg]
    else:
        if arg and os.path.isdir(arg):
            search_dirs.append(arg)
        search_dirs += [os.getcwd(), here, "/mnt/user-data/uploads"]
        seen = set()                      # a file name is used once, even if present in several folders
        res_real = os.path.realpath(RESULTS_DIR)
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for ext in ("*.xlsx", "*.xls", "*.csv"):
                for f in sorted(glob.glob(os.path.join(d, ext))):
                    rp = os.path.realpath(f)
                    base = os.path.basename(f)
                    if base.lower() in seen or base.startswith("~$") or rp.startswith(res_real):
                        continue
                    seen.add(base.lower())
                    files.append(f)
    if not files:
        print("ERROR: no .xlsx / .xls / .csv files found.")
        print("Place the four datasets next to this script, or run:  python script.py path/to/folder")
        sys.exit(1)

    per_file = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == ".csv":
                tables = [(None, read_csv_any(f))]
            else:
                xl = pd.ExcelFile(f)
                tables = []
                for sh in xl.sheet_names:                       # inspect EVERY sheet
                    try:
                        tables.append((sh, xl.parse(sh)))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  (skipped unreadable file {os.path.basename(f)}: {e})")
            continue
        cands = []
        for sh, df in tables:
            mp = match_columns(df.columns)
            if set(mp.values()) >= set(REQUIRED_COLUMNS):
                d2 = df.rename(columns=mp)
                cands.append({"path": f, "sheet": sh, "df": d2, "rows": len(d2),
                              "valid": int(d2[TARGET].notna().sum())})
        if not cands:
            print(f"  (no sheet of {os.path.basename(f)} contains the required columns "
                  f"{REQUIRED_COLUMNS}; file ignored)")
            continue
        # prefer an explicitly 'clean' sheet, otherwise the sheet with most labelled rows
        cands.sort(key=lambda c: (0 if (c["sheet"] and "clean" in str(c["sheet"]).lower()) else 1, -c["valid"]))
        best = cands[0]
        best["other_sheets"] = [c["sheet"] for c in cands[1:]]
        per_file.append(best)

    if not per_file:
        print("ERROR: none of the files contains the required columns.")
        print(f"Required columns: {REQUIRED_COLUMNS}")
        sys.exit(1)

    per_file.sort(key=lambda c: os.path.basename(c["path"]).lower())
    if len(per_file) > N_DATASETS:
        print(f"WARNING: {len(per_file)} dataset files found; using the first {N_DATASETS} (sorted by file name). "
              "Ignored: " + ", ".join(os.path.basename(c["path"]) for c in per_file[N_DATASETS:]))
        per_file = per_file[:N_DATASETS]
    elif len(per_file) < N_DATASETS:
        print(f"WARNING: only {len(per_file)} dataset file(s) with the required columns were found "
              f"(expected {N_DATASETS}). The analysis continues with the datasets that were found.")
    return per_file


# --------------------------------------------------------------------------
# 2. Preprocessing (fit on the training set only)
# --------------------------------------------------------------------------
class Preprocessor:
    """Median imputation -> IQR outlier clipping -> standard scaling.
    Every learned parameter is fitted on the TRAINING data only."""

    def __init__(self, iqr_factor=OUTLIER_IQR_FACTOR):
        self.iqr_factor = iqr_factor
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X):
        Xi = self.imputer.fit_transform(X.values.astype(float))
        q1 = np.percentile(Xi, 25, axis=0)
        q3 = np.percentile(Xi, 75, axis=0)
        iqr = q3 - q1
        self.lower = np.where(iqr > 0, q1 - self.iqr_factor * iqr, -np.inf)
        self.upper = np.where(iqr > 0, q3 + self.iqr_factor * iqr, np.inf)
        self.scaler.fit(np.clip(Xi, self.lower, self.upper))
        return self

    def n_clipped(self, X):
        Xi = self.imputer.transform(X.values.astype(float))
        return ((Xi < self.lower) | (Xi > self.upper)).sum(axis=0)

    def transform(self, X):
        Xi = self.imputer.transform(np.asarray(X, dtype=float))
        Xi = np.clip(Xi, self.lower, self.upper)
        return self.scaler.transform(Xi)


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
class LMRRegressionBaseline:
    """Linear (multi-linear) regression on ordinal class codes; predictions are rounded and
    clipped to valid class codes so that classification metrics can be computed."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y.astype(float))
        return self

    def predict(self, X):
        p = np.rint(self.model.predict(X))
        return np.clip(p, 0, self.n_classes - 1).astype(int)


def build_models(n_classes):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "LMR": LMRRegressionBaseline(n_classes),
        "XGB": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             objective="multi:softprob", eval_metric="mlogloss",
                             tree_method="hist", random_state=RANDOM_STATE,
                             n_jobs=-1, verbosity=0),
        "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }


# --------------------------------------------------------------------------
# 4. Metrics (percent, unrounded)
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    kw = dict(zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, average="weighted", **kw) * 100,
        "Recall": recall_score(y_true, y_pred, average="weighted", **kw) * 100,
        "F1": f1_score(y_true, y_pred, average="weighted", **kw) * 100,
        "Macro Precision": precision_score(y_true, y_pred, average="macro", **kw) * 100,
        "Macro Recall": recall_score(y_true, y_pred, average="macro", **kw) * 100,
        "Macro F1": f1_score(y_true, y_pred, average="macro", **kw) * 100,
    }


# --------------------------------------------------------------------------
# 5. Plot helpers (all take an explicit output path)
# --------------------------------------------------------------------------
def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_bar_by_model(df_split, metric, ylabel, title, path):
    d = df_split.set_index("Model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODELS, d[metric].values, color=[MODEL_COLORS[m] for m in MODELS],
                  edgecolor="black", width=0.6)
    for b, v in zip(bars, d[metric].values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}%", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


def plot_across_splits(perf, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.25
    x = np.arange(len(MODELS))
    for i, lab in enumerate(SPLIT_LABELS):
        vals = [perf[(perf.Split == lab) & (perf.Model == m)][metric].values[0] for m in MODELS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=SPLIT_COLORS[lab], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Train/Test split", loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


def plot_precision_recall(df_split, label, ds_label, path):
    d = df_split.set_index("Model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in MODELS:
        ax.scatter(d.loc[m, "Recall"], d.loc[m, "Precision"], s=160, color=MODEL_COLORS[m],
                   edgecolor="black", zorder=3, label=m)
    lo = min(d["Recall"].min(), d["Precision"].min())
    hi = max(d["Recall"].max(), d["Precision"].max())
    pad = max((hi - lo) * 0.25, 1.5)
    ax.set_xlim(lo - pad, min(hi + pad, 101))
    ax.set_ylim(lo - pad, min(hi + pad, 101))
    offs = {"LR": (10, 10), "LMR": (10, -16), "XGB": (-12, 12), "RF": (-12, -18)}
    for m in MODELS:
        ax.annotate(f"{m} ({d.loc[m, 'Recall']:.2f}, {d.loc[m, 'Precision']:.2f})",
                    (d.loc[m, "Recall"], d.loc[m, "Precision"]), textcoords="offset points",
                    xytext=offs[m], fontsize=9, fontweight="bold",
                    ha="left" if offs[m][0] > 0 else "right")
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(f"{ds_label}: Precision vs Recall (weighted) - {label} split")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    save_fig(fig, path)


def plot_mean_with_std(summary, mean_col, std_col, ylabel, title, path):
    d = summary.set_index("Model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODELS, d[mean_col].values, yerr=d[std_col].values, capsize=6,
                  color=[MODEL_COLORS[m] for m in MODELS], edgecolor="black", width=0.6)
    for b, v, s in zip(bars, d[mean_col].values, d[std_col].values):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.8, f"{v:.2f}%", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


def plot_robustness(perf, rob, metric, ylabel, title, path, split_label=None):
    """Lines per model. split_label=None -> all splits (line style = split).
    The 0 % point is the unperturbed result of the same fixed model."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = SPLIT_LABELS if split_label is None else [split_label]
    for lab in labels:
        for m in MODELS:
            base = perf[(perf.Split == lab) & (perf.Model == m)][metric].values[0]
            r = rob[(rob.Split == lab) & (rob.Model == m)].sort_values("Perturbation")
            xs = [0] + list(r["Perturbation"].values)
            ys = [base] + list(r[metric].values)
            ax.plot(xs, ys, linestyle=SPLIT_STYLES[lab], marker="o", color=MODEL_COLORS[m],
                    linewidth=2, label=m if split_label else f"{m} ({lab})")
    ax.set_xticks([0, 20, 30, 40])
    ax.set_xticklabels(["0 (clean)", "20", "30", "40"])
    ax.set_xlabel("Perturbation Level (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=1 if split_label else 3, fontsize=9 if split_label else 8, loc="lower left")
    save_fig(fig, path)


def plot_confusion(cm, class_names, model, label, ds_label, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    thr = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thr else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{ds_label}: Confusion Matrix - {model} ({label} split)\n(number of test samples)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, path)


# --------------------------------------------------------------------------
# 6. ONE complete, independent analysis of ONE dataset
# --------------------------------------------------------------------------
def analyze_dataset(k, cand):
    tag = f"Dataset{k}"
    label = f"Dataset {k}"
    root = os.path.join(RESULTS_DIR, f"Dataset_{k}")
    gdir = os.path.join(root, "graphs")
    cdir = os.path.join(root, "confusion_matrices")
    vdir = os.path.join(root, "CSV")
    for d in (gdir, cdir, vdir):
        os.makedirs(d, exist_ok=True)

    def gp(name):
        return os.path.join(gdir, f"{tag}_{name}")

    raw = cand["df"]
    file_name = os.path.basename(cand["path"])
    sheet_name = cand["sheet"]

    banner(f"{label.upper()} - {file_name}")
    print(f"Filename           : {file_name}")
    print(f"Selected sheet     : {sheet_name if sheet_name else 'N/A (CSV file)'}")
    if cand.get("other_sheets"):
        print(f"Other valid sheets : {cand['other_sheets']} (not used)")
    print(f"Number of rows     : {raw.shape[0]}")
    print(f"Number of columns  : {raw.shape[1]}")
    print(f"Columns            : {list(raw.columns)}")

    # ---------------- cleaning (no learned parameters here) ----------------
    n_raw = len(raw)
    df = raw.dropna(how="all")
    n_empty = n_raw - len(df)
    n_before_dup = len(df)
    df = df.drop_duplicates()
    n_dups = n_before_dup - len(df)

    data = df[REQUIRED_COLUMNS].copy()          # only the required columns are kept
    n_invalid_num = {}
    for c in REQUIRED_FEATURES:
        before_na = int(data[c].isna().sum())
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].replace([np.inf, -np.inf], np.nan)
        n_invalid_num[c] = int(data[c].isna().sum()) - before_na
    data[TARGET] = data[TARGET].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    data.loc[data[TARGET].isin(["", "nan", "None"]), TARGET] = np.nan
    n_missing_target = int(data[TARGET].isna().sum())
    data = data[data[TARGET].notna()].reset_index(drop=True)
    missing_feat = {c: int(data[c].isna().sum()) for c in REQUIRED_FEATURES}
    n_samples = len(data)

    print(f"Empty rows removed : {n_empty} | duplicate rows removed: {n_dups} | "
          f"rows without {TARGET} removed: {n_missing_target}")
    print(f"Samples before preprocessing: {n_raw} | after preprocessing: {n_samples}")
    print(f"Invalid numeric values -> NaN: {n_invalid_num} | missing feature values (imputed later): {missing_feat}")

    # ---------------- target classes ----------------
    known_order = {"low": 0, "moderate": 1, "medium": 1, "high": 2, "veryhigh": 3, "very high": 3}
    names_all = sorted(data[TARGET].unique())
    if all(c.lower() in known_order for c in names_all):
        class_names = sorted(names_all, key=lambda c: known_order[c.lower()])
        order_basis = "known risk ordering of the class names (Low < Moderate < High)"
    else:
        class_names = list(data.groupby(TARGET)["AVG_NOISE_LVL"].mean().sort_values().index)
        order_basis = "ascending mean AVG_NOISE_LVL of each class"
    class_to_code = {c: i for i, c in enumerate(class_names)}
    y_all = data[TARGET].map(class_to_code).astype(int).values
    n_classes = len(class_names)
    X_all = data[REQUIRED_FEATURES].copy()

    dist = data[TARGET].value_counts().reindex(class_names)
    class_table = pd.DataFrame({"Class": class_names,
                                "Number of Samples": dist.values.astype(int),
                                "Percentage": dist.values / dist.values.sum() * 100})
    print(f"Target classes (ordinal order): {class_names}")
    print(class_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    class_range = data.groupby(TARGET)["AVG_NOISE_LVL"].agg(["min", "max"]).reindex(class_names)
    rv = class_range.values
    classes_separable = all(rv[i, 1] < rv[i + 1, 0] for i in range(n_classes - 1))
    constant_features = [c for c in REQUIRED_FEATURES if X_all[c].nunique(dropna=True) <= 1]

    # ---------------- experiments ----------------
    perf_rows, rob_rows, outlier_rows = [], [], []
    primary_cm = {}
    for si, (test_size, lab) in enumerate(SPLITS):
        try:
            Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=test_size,
                                                  random_state=RANDOM_STATE, stratify=y_all)
        except ValueError:
            print(f"  (stratification not possible for {lab}; non-stratified split used)")
            Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=test_size,
                                                  random_state=RANDOM_STATE)
        print(f"\n--- {label} | split {lab}: train={len(Xtr)}, test={len(Xte)} ---")

        prep = Preprocessor().fit(Xtr)                 # TRAIN only
        Xtr_p, Xte_p = prep.transform(Xtr), prep.transform(Xte)
        clip_tr, clip_te = prep.n_clipped(Xtr), prep.n_clipped(Xte)
        for j, f in enumerate(REQUIRED_FEATURES):
            outlier_rows.append({"Split": lab, "Feature": f, "Lower bound": prep.lower[j],
                                 "Upper bound": prep.upper[j], "Clipped (train)": int(clip_tr[j]),
                                 "Clipped (test)": int(clip_te[j])})

        models = build_models(n_classes)
        for m in MODELS:
            models[m].fit(Xtr_p, ytr)

        for m in MODELS:
            pred = models[m].predict(Xte_p)
            res = evaluate(yte, pred)
            perf_rows.append({"Split": lab, "Model": m, **res})
            if lab == PRIMARY_SPLIT:
                primary_cm[m] = confusion_matrix(yte, pred, labels=list(range(n_classes)))
            print(f"  {m:<4} Acc={res['Accuracy']:.2f}%  P={res['Precision']:.2f}%  "
                  f"R={res['Recall']:.2f}%  F1={res['F1']:.2f}%")

        # robustness: trained models stay FIXED, only the test features are perturbed
        train_std = np.nan_to_num(np.nanstd(Xtr.values.astype(float), axis=0, ddof=1), nan=0.0)
        Xte_raw = Xte.values.astype(float)
        for level in PERTURBATIONS:
            rng = np.random.default_rng(RANDOM_STATE + 1000 * si + level)
            noise = rng.standard_normal(Xte_raw.shape)
            Xte_pert = Xte_raw + noise * train_std * (level / 100.0)
            Xte_pert_p = prep.transform(Xte_pert)
            for m in MODELS:
                res = evaluate(yte, models[m].predict(Xte_pert_p))
                rob_rows.append({"Split": lab, "Model": m, "Perturbation": level,
                                 "Accuracy": res["Accuracy"], "Precision": res["Precision"],
                                 "Recall": res["Recall"], "F1": res["F1"]})

    perf = pd.DataFrame(perf_rows)
    rob = pd.DataFrame(rob_rows)
    outlier_df = pd.DataFrame(outlier_rows)

    summary_rows = []
    for m in MODELS:
        d = perf[perf.Model == m]
        summary_rows.append({
            "Model": m,
            "Mean Accuracy": d["Accuracy"].mean(), "Std Accuracy": d["Accuracy"].std(ddof=1),
            "Mean Precision": d["Precision"].mean(), "Std Precision": d["Precision"].std(ddof=1),
            "Mean Recall": d["Recall"].mean(), "Std Recall": d["Recall"].std(ddof=1),
            "Mean F1": d["F1"].mean(), "Std F1": d["F1"].std(ddof=1)})
    summary = pd.DataFrame(summary_rows)

    # ---------------- CSV ----------------
    perf.to_csv(os.path.join(vdir, f"{tag}_model_performance.csv"), index=False)
    rob.to_csv(os.path.join(vdir, f"{tag}_robustness_results.csv"), index=False)
    summary.to_csv(os.path.join(vdir, f"{tag}_model_summary.csv"), index=False)

    # ---------------- graphs ----------------
    fig = {"acc": {}, "f1": {}, "pr": {}, "cm": {}}
    for lab in SPLIT_LABELS:
        d = perf[perf.Split == lab]
        t = split_tag(lab)
        fig["acc"][lab] = gp(f"accuracy_vs_model_{t}.png")
        fig["f1"][lab] = gp(f"f1_vs_model_{t}.png")
        fig["pr"][lab] = gp(f"precision_vs_recall_{t}.png")
        plot_bar_by_model(d, "Accuracy", "Accuracy (%)", f"{label}: Accuracy vs Model - {lab} split", fig["acc"][lab])
        plot_bar_by_model(d, "F1", "Weighted F1 Score (%)", f"{label}: Weighted F1 vs Model - {lab} split", fig["f1"][lab])
        plot_precision_recall(d, lab, label, fig["pr"][lab])
    fig["acc_across"] = gp("accuracy_across_splits.png")
    fig["f1_across"] = gp("f1_across_splits.png")
    plot_across_splits(perf, "Accuracy", "Accuracy (%)", f"{label}: Accuracy across train/test splits", fig["acc_across"])
    plot_across_splits(perf, "F1", "Weighted F1 Score (%)", f"{label}: Weighted F1 across train/test splits", fig["f1_across"])

    fig["rob_acc"] = gp("robustness_accuracy.png")
    fig["rob_f1"] = gp("robustness_f1.png")
    fig["rob_acc_all"] = gp("robustness_accuracy_all_splits.png")
    fig["rob_f1_all"] = gp("robustness_f1_all_splits.png")
    plot_robustness(perf, rob, "Accuracy", "Accuracy (%)",
                    f"{label}: Robustness - Accuracy ({PRIMARY_SPLIT} split)", fig["rob_acc"], PRIMARY_SPLIT)
    plot_robustness(perf, rob, "F1", "Weighted F1 Score (%)",
                    f"{label}: Robustness - Weighted F1 ({PRIMARY_SPLIT} split)", fig["rob_f1"], PRIMARY_SPLIT)
    plot_robustness(perf, rob, "Accuracy", "Accuracy (%)",
                    f"{label}: Robustness - Accuracy (all splits)", fig["rob_acc_all"])
    plot_robustness(perf, rob, "F1", "Weighted F1 Score (%)",
                    f"{label}: Robustness - Weighted F1 (all splits)", fig["rob_f1_all"])

    for m in MODELS:
        fig["cm"][m] = os.path.join(cdir, f"{tag}_confusion_matrix_{m}.png")
        plot_confusion(primary_cm[m], class_names, m, PRIMARY_SPLIT, label, fig["cm"][m])

    fig["mean_acc"] = gp("mean_accuracy_by_model.png")
    fig["mean_f1"] = gp("mean_f1_by_model.png")
    plot_mean_with_std(summary, "Mean Accuracy", "Std Accuracy", "Mean Accuracy (%)",
                       f"{label}: Mean Accuracy by Model (error bars = std)", fig["mean_acc"])
    plot_mean_with_std(summary, "Mean F1", "Std F1", "Mean Weighted F1 (%)",
                       f"{label}: Mean Weighted F1 by Model (error bars = std)", fig["mean_f1"])

    # ---------------- best model (computed, never chosen manually) ----------------
    best_by_split = {}
    for lab in SPLIT_LABELS:
        d = perf[perf.Split == lab].sort_values(["F1", "Accuracy"], ascending=False).reset_index(drop=True)
        ds = d.set_index("Model")
        tied = [mm for mm in d["Model"][1:] if abs(ds.loc[mm, "F1"] - d.loc[0, "F1"]) < 1e-9
                and abs(ds.loc[mm, "Accuracy"] - d.loc[0, "Accuracy"]) < 1e-9]
        name = d.loc[0, "Model"] + (f" (tied with {', '.join(tied)})" if tied else "")
        best_by_split[lab] = (name, d.loc[0, "F1"], d.loc[0, "Accuracy"])

    sc = "Consistency Score (Mean F1 - Std F1)"
    summary[sc] = summary["Mean F1"] - summary["Std F1"]
    cons = summary.sort_values([sc, "Mean F1"], ascending=False).reset_index(drop=True)
    cs = cons.set_index("Model")
    ctied = [mm for mm in cons["Model"][1:] if abs(cs.loc[mm, sc] - cons.loc[0, sc]) < 1e-9
             and abs(cs.loc[mm, "Mean F1"] - cons.loc[0, "Mean F1"]) < 1e-9]
    most_consistent = cons.loc[0, "Model"] + (f" (tied with {', '.join(ctied)})" if ctied else "")

    rob_sum_rows = []
    for m in MODELS:
        base = perf[perf.Model == m]["F1"].mean()
        row = {"Model": m, "Mean F1 @0% (clean)": base}
        for lv in PERTURBATIONS:
            row[f"Mean F1 @{lv}%"] = rob[(rob.Model == m) & (rob.Perturbation == lv)]["F1"].mean()
        row["F1 drop 0%->40%"] = base - row["Mean F1 @40%"]
        rob_sum_rows.append(row)
    rob_sum = pd.DataFrame(rob_sum_rows)

    print(f"\n{label} best model per split: " +
          ", ".join(f"{l}: {best_by_split[l][0]}" for l in SPLIT_LABELS))
    print(f"{label} most consistent model: {most_consistent}")

    return {
        "k": k, "tag": tag, "label": label, "file": file_name, "sheet": sheet_name,
        "n_raw": n_raw, "n_cols": raw.shape[1], "columns": list(raw.columns),
        "n_empty": n_empty, "n_dups": n_dups, "n_missing_target": n_missing_target,
        "n_samples": n_samples, "missing_feat": missing_feat, "n_invalid_num": n_invalid_num,
        "class_names": class_names, "class_to_code": class_to_code, "order_basis": order_basis,
        "class_table": class_table, "class_range": class_range,
        "classes_separable": classes_separable, "constant_features": constant_features,
        "perf": perf, "rob": rob, "summary": summary, "cons": cons, "rob_sum": rob_sum,
        "outlier_df": outlier_df, "best_by_split": best_by_split, "most_consistent": most_consistent,
        "fig": fig, "n_classes": n_classes,
    }


# --------------------------------------------------------------------------
# 7. Cross-dataset comparison (summaries of the stored results only)
# --------------------------------------------------------------------------
def ds_tick_labels(results):
    import textwrap
    return [f"{r['label']}\n" + "\n".join(textwrap.wrap(r["file"].replace("_", " "), 20)) for r in results]


def plot_overall_metric(results, all_perf, metric, path):
    labels = ds_tick_labels(results)
    x = np.arange(len(results))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(MODELS):
        vals = [all_perf[(all_perf.Dataset == r["label"]) & (all_perf.Model == m)][metric].mean() for r in results]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=MODEL_COLORS[m], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_ylabel(f"Mean {metric} over 80/20, 70/30, 60/40 (%)")
    ax.set_title(f"Overall {metric} comparison across datasets (each dataset trained and tested separately)")
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


def plot_overall_robustness(results, path):
    n = len(results)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.3 * rows), squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        for m in MODELS:
            base = r["perf"][r["perf"].Model == m]["F1"].mean()
            ys = [base] + [r["rob"][(r["rob"].Model == m) & (r["rob"].Perturbation == lv)]["F1"].mean()
                           for lv in PERTURBATIONS]
            ax.plot([0] + PERTURBATIONS, ys, marker="o", linewidth=2, color=MODEL_COLORS[m], label=m)
        ax.set_xticks([0, 20, 30, 40])
        ax.set_title(f"{r['label']} ({r['file']})", fontsize=8)
        ax.set_xlabel("Perturbation Level (%)  [0 = clean]")
        ax.set_ylabel("Mean weighted F1 (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("Robustness comparison: mean weighted F1 over the three splits", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_overall_consistency(results, all_summary, path):
    labels = ds_tick_labels(results)
    x = np.arange(len(results))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(MODELS):
        vals = [all_summary[(all_summary.Dataset == r["label"]) & (all_summary.Model == m)]["Std F1"].values[0]
                for r in results]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=MODEL_COLORS[m], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Std of weighted F1 across the three splits (percentage points)")
    ax.set_title("Model consistency comparison (lower = more consistent)")
    ax.margins(y=0.2)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


# --------------------------------------------------------------------------
# 8. PDF helpers
# --------------------------------------------------------------------------
def make_styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=ss["Title"], fontSize=22, leading=28, alignment=TA_CENTER,
                                spaceAfter=14, textColor=colors.HexColor("#1a2a4a")),
        "h1": ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, leading=21, spaceBefore=4,
                             spaceAfter=10, textColor=colors.HexColor("#1a2a4a")),
        "h2": ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=17, spaceBefore=10,
                             spaceAfter=6, textColor=colors.HexColor("#2c4a7a")),
        "h3": ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, leading=14, spaceBefore=6,
                             spaceAfter=4, textColor=colors.HexColor("#2c4a7a")),
        "body": ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14.5,
                               alignment=TA_JUSTIFY, spaceAfter=6),
        "caption": ParagraphStyle("C", parent=ss["BodyText"], fontSize=8.5, leading=11, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=10),
        "cell": ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8, leading=10),
        "cellh": ParagraphStyle("CellH", parent=ss["BodyText"], fontSize=8, leading=10,
                                textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER),
        "sub": ParagraphStyle("S", parent=ss["BodyText"], fontSize=12, leading=16, alignment=TA_CENTER,
                              textColor=colors.HexColor("#444444")),
        "tocT": ParagraphStyle("TocT", parent=ss["Heading1"], fontSize=17, leading=21, spaceAfter=10,
                               textColor=colors.HexColor("#1a2a4a")),
    }


def P(text, style):
    return Paragraph(text, style)


def df_table(df, st, col_widths=None, decimals=2):
    rows = [[Paragraph(escape(str(c)), st["cellh"]) for c in df.columns]]
    for _, r in df.iterrows():
        row = []
        for v in r.values:
            s = f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)
            row.append(Paragraph(escape(s), st["cell"]))
        rows.append(row)
    t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c4a7a")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def fig_block(path, caption, st, max_w=15.5 * cm, max_h=15 * cm):
    iw, ih = ImageReader(path).getSize()
    scale = min(max_w / iw, max_h / ih)
    img = Image(path, width=iw * scale, height=ih * scale)
    img.hAlign = "CENTER"
    return KeepTogether([img, P(escape(caption), st["caption"])])


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"Page {doc.page}")
    canvas.drawString(2 * cm, 1.1 * cm, "Road Noise Risk Level Prediction - Four-Dataset Analysis")
    canvas.restoreState()


class ReportDoc(SimpleDocTemplate):
    """Registers H1/H2 headings for the table of contents."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in ("H1", "H2"):
                level = 0 if name == "H1" else 1
                key = f"toc_{id(flowable)}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))


def conclusion_points(R):
    perf, summary = R["perf"], R["summary"]
    rs = R["rob_sum"]
    ranked = summary.sort_values(["Mean F1", "Std F1"], ascending=[False, True]).reset_index(drop=True)
    top, bottom = ranked.loc[0], ranked.loc[len(ranked) - 1]
    winners = [R["best_by_split"][l][0] for l in SPLIT_LABELS]
    if len(set(winners)) == 1:
        win_txt = f"{winners[0]} achieved the highest weighted F1 in all three splits."
        if "tied" in winners[0]:
            win_txt += " Tied models obtained identical accuracy and weighted F1."
    else:
        win_txt = "The best model per split was " + ", ".join(
            f"{R['best_by_split'][l][0]} ({l})" for l in SPLIT_LABELS) + "."
    top_rows = perf[perf.Model == top["Model"]]
    pr_gap = (top_rows["Precision"] - top_rows["Recall"]).abs().max()
    std_max = summary["Std F1"].max()
    most_rob = rs.sort_values("F1 drop 0%->40%").iloc[0]
    least_rob = rs.sort_values("F1 drop 0%->40%").iloc[-1]
    pts = [
        f"On {R['label']} the highest mean weighted F1 (ties resolved by lower standard deviation) was "
        f"obtained by {top['Model']} ({top['Mean F1']:.2f}%, mean accuracy {top['Mean Accuracy']:.2f}%), and the "
        f"lowest by {bottom['Model']} ({bottom['Mean F1']:.2f}%). {win_txt}",
        f"Weighted F1 ranged from {perf['F1'].min():.2f}% to {perf['F1'].max():.2f}% over all models and splits. "
        f"The largest standard deviation of F1 across splits was {std_max:.3f} percentage points, so changing the "
        "train/test proportion "
        + ("had only a small effect." if std_max < 1.0 else "noticeably changed the results of at least one model."),
        f"For {top['Model']}, weighted precision and recall differed by at most {pr_gap:.2f} percentage points "
        "within any split, i.e. "
        + ("balanced" if pr_gap < 1.0 else "somewhat unbalanced") + " precision/recall behaviour.",
        f"Under test-feature perturbation (mean over the three splits), {most_rob['Model']} lost the least weighted F1 "
        f"between clean data and 40% noise ({most_rob['F1 drop 0%->40%']:.2f} percentage points), and "
        f"{least_rob['Model']} lost the most ({least_rob['F1 drop 0%->40%']:.2f} percentage points).",
        f"Considering mean performance and variability across splits, {R['most_consistent']} is the most consistent "
        "model on this dataset.",
    ]
    if R["classes_separable"]:
        pts.append("The AVG_NOISE_LVL ranges of the risk classes do not overlap in this dataset, so high scores mainly "
                   "reflect how well each model recovers the noise-level thresholds.")
    return pts


def build_dataset_section(S, R, st):
    k, label = R["k"], R["label"]
    fg = R["fig"]
    perf, rob = R["perf"], R["rob"]
    cls = ", ".join(R["class_names"])
    feats = ", ".join(REQUIRED_FEATURES)

    S.append(PageBreak())
    S.append(P(escape(f"{label}: {R['file']}"), st["h1"]))

    # 1 Introduction
    S.append(P(escape(f"1. {label} Introduction"), st["h2"]))
    S.append(P(
        f"This section reports the complete, independent machine-learning analysis of <b>{label}</b> "
        f"(file <b>{escape(R['file'])}</b>). The aim is to predict the road-noise risk level "
        f"(<b>{TARGET}</b>; classes: {escape(cls)}) from four numerical measurement attributes. Four models "
        "(LR, LMR, XGB, RF) are trained and evaluated under three train/test splits (80/20, 70/30, 60/40), "
        "and the trained models are then stress-tested with Gaussian perturbation of the test features at "
        "20%, 30% and 40%. All preprocessing, training and evaluation use only the data of this dataset; "
        "no data from any other dataset is used.", st["body"]))

    # 2 Description
    S.append(P(escape(f"2. {label} Description"), st["h2"]))
    info = [
        ["Filename", R["file"]],
        ["Selected sheet", str(R["sheet"]) if R["sheet"] else "N/A (CSV file)"],
        ["Rows in file", str(R["n_raw"])],
        ["Columns in file", f"{R['n_cols']}"],
        ["Rows used after preprocessing", str(R["n_samples"])],
        ["Features", feats],
        ["Target", TARGET],
    ]
    t = Table([[Paragraph(f"<b>{escape(a)}</b>", st["cell"]), Paragraph(escape(b), st["cell"])] for a, b in info],
              colWidths=[5 * cm, 11.5 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S.append(t)
    S.append(Spacer(1, 6))
    S.append(P(f"Columns of the file: {escape(', '.join(map(str, R['columns'])))}. Identifier, town, coordinate, "
               "date and text columns are deliberately not used as features.", st["body"]))
    S.append(P("<b>Class distribution</b> (classes in ordinal order):", st["body"]))
    S.append(df_table(R["class_table"], st, col_widths=[5 * cm, 4.5 * cm, 4 * cm]))
    S.append(P(f"Table: class distribution of {label}.", st["caption"]))
    rng_txt = "; ".join(f"{c}: {R['class_range'].loc[c, 'min']:.1f} to {R['class_range'].loc[c, 'max']:.1f} dB"
                        for c in R["class_names"])
    S.append(P(f"Observed AVG_NOISE_LVL range per class - {escape(rng_txt)}.", st["body"]))
    if R["classes_separable"]:
        S.append(P("The AVG_NOISE_LVL ranges of the classes do not overlap, i.e. the label can be separated by "
                   "thresholds on AVG_NOISE_LVL alone; very high scores are therefore expected.", st["body"]))
    if R["constant_features"]:
        S.append(P(f"Constant feature(s) with a single unique value (no predictive information): "
                   f"{escape(', '.join(R['constant_features']))}.", st["body"]))

    # 3 Preprocessing
    S.append(P(escape(f"3. {label} Preprocessing"), st["h2"]))
    S.append(P(
        f"<b>Samples before / after preprocessing:</b> {R['n_raw']} / {R['n_samples']}. "
        f"Removed: {R['n_empty']} completely empty rows, {R['n_dups']} duplicate rows and "
        f"{R['n_missing_target']} rows without a {TARGET} value (unusable for supervised learning). "
        f"Missing feature values after cleaning: {escape(str(R['missing_feat']))}.", st["body"]))
    S.append(P("<b>Missing and invalid values.</b> Feature columns are converted to numeric; invalid or infinite "
               "values are treated as missing and imputed with the training-set median.", st["body"]))
    S.append(P(f"<b>Outlier handling.</b> IQR winsorization: values outside [Q1 - {OUTLIER_IQR_FACTOR:g}xIQR, "
               f"Q3 + {OUTLIER_IQR_FACTOR:g}xIQR] are clipped (not deleted). The bounds are computed on the "
               "training set only and applied to training and test data, so no sample is lost.", st["body"]))
    code_txt = ", ".join("%s=%d" % (c, R["class_to_code"][c]) for c in R["class_names"])
    S.append(P(f"<b>Target encoding.</b> {TARGET} is encoded as ordinal codes "
               f"({escape(code_txt)}); the order follows the "
               f"{escape(R['order_basis'])}. Original class names are kept for tables and confusion matrices.", st["body"]))
    S.append(P("<b>Scaling and leakage prevention.</b> StandardScaler is fitted on the training features only. For "
               f"every split (stratified, random_state = {RANDOM_STATE}) the data are split first, then the "
               "imputer, outlier bounds and scaler are fitted on the training portion and applied to the test portion.",
               st["body"]))
    out_show = R["outlier_df"].copy()
    out_show["Lower bound"] = out_show["Lower bound"].map(lambda v: "-" if np.isinf(v) else f"{v:.2f}")
    out_show["Upper bound"] = out_show["Upper bound"].map(lambda v: "-" if np.isinf(v) else f"{v:.2f}")
    S.append(df_table(out_show, st, col_widths=[2 * cm, 3.6 * cm, 2.6 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm]))
    S.append(P(f"Table: outlier-clipping bounds (training set) and number of clipped values, {label}.", st["caption"]))

    # 4 Models
    S.append(P(escape(f"4. {label} Models"), st["h2"]))
    S.append(P("<b>LR</b> - multinomial Logistic Regression (max_iter=2000, random_state=42).", st["body"]))
    S.append(P("<b>LMR</b> - Linear/Multi-linear Regression <i>baseline</i>. Ordinary least squares is trained on the "
               "ordinal class codes and predicts a continuous value; the value is rounded to the nearest integer and "
               "clipped to the valid code range, and classification metrics are computed from these converted labels. "
               "It is a regression baseline, not a native multiclass classifier.", st["body"]))
    S.append(P("<b>XGB</b> - XGBoost classifier (200 trees, max_depth=4, learning_rate=0.1, multi:softprob, hist, "
               "random_state=42).", st["body"]))
    S.append(P("<b>RF</b> - Random Forest classifier (200 trees, random_state=42).", st["body"]))
    S.append(P("Metrics are Accuracy and weighted Precision, Recall and F1 (macro averages are stored in the CSV "
               "files); all are shown in percent, rounded only for display.", st["body"]))

    # 5-7 split results
    for n, lab in enumerate(SPLIT_LABELS, start=5):
        S.append(PageBreak())
        S.append(P(escape(f"{n}. {label} - {lab} Results"), st["h2"]))
        d = perf[perf.Split == lab][["Model", "Accuracy", "Precision", "Recall", "F1"]]
        S.append(df_table(d, st, col_widths=[3 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm]))
        S.append(P(f"Table: {label} - {lab}: accuracy and weighted precision, recall and F1 (%).", st["caption"]))
        bs = R["best_by_split"][lab]
        S.append(P(f"Best model in this split (highest weighted F1): <b>{escape(bs[0])}</b> "
                   f"(F1 = {bs[1]:.2f}%, accuracy = {bs[2]:.2f}%).", st["body"]))
        S.append(fig_block(fg["acc"][lab], f"Figure: {label} - accuracy (%) of each model, {lab} split.", st, 14 * cm, 8.2 * cm))
        S.append(fig_block(fg["f1"][lab], f"Figure: {label} - weighted F1 (%) of each model, {lab} split.", st, 14 * cm, 8.2 * cm))
        S.append(fig_block(fg["pr"][lab], f"Figure: {label} - precision vs recall, {lab} split.", st, 14 * cm, 9 * cm))

    # 8 Robustness
    S.append(PageBreak())
    S.append(P(escape(f"8. {label} - Robustness"), st["h2"]))
    S.append(P(
        f"Each model is trained once on clean training data and kept fixed. Gaussian noise (perturbed = original + "
        f"N(0,1) x training-set feature std x perturbation level) is added to the raw test features at 20%, 30% and "
        f"40%; the noisy features pass through the already-fitted preprocessing and are scored by the fixed models. "
        f"The target is not modified and nothing is retrained. The <b>{PRIMARY_SPLIT}</b> split is the primary "
        "robustness experiment (graphs); the same test was also run for 70/30 and 60/40 and is included in the "
        "tables and in the all-splits graphs. Perturbation percentages are unrelated to the split percentages.", st["body"]))
    rs = rob[rob.Split == PRIMARY_SPLIT][["Model", "Perturbation", "Accuracy", "Precision", "Recall", "F1"]].copy()
    rs["Perturbation"] = rs["Perturbation"].map(lambda v: f"{v}%")
    S.append(P(f"<b>Primary experiment ({PRIMARY_SPLIT})</b>", st["h3"]))
    S.append(df_table(rs, st, col_widths=[2.6 * cm, 3 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]))
    S.append(P(f"Table: {label} - robustness results (%), {PRIMARY_SPLIT} split.", st["caption"]))
    S.append(fig_block(fg["rob_acc"], f"Figure: {label} - accuracy vs perturbation level ({PRIMARY_SPLIT} split; 0 = clean).", st, 14 * cm, 9 * cm))
    S.append(fig_block(fg["rob_f1"], f"Figure: {label} - weighted F1 vs perturbation level ({PRIMARY_SPLIT} split; 0 = clean).", st, 14 * cm, 9 * cm))
    S.append(P("<b>All splits</b>", st["h3"]))
    ra = rob[["Split", "Model", "Perturbation", "Accuracy", "Precision", "Recall", "F1"]].copy()
    ra["Perturbation"] = ra["Perturbation"].map(lambda v: f"{v}%")
    S.append(df_table(ra, st, col_widths=[2.2 * cm, 2.4 * cm, 2.8 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    S.append(P(f"Table: {label} - robustness results (%) for all splits.", st["caption"]))
    S.append(df_table(R["rob_sum"], st, col_widths=[2 * cm, 2.9 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm]))
    S.append(P(f"Table: {label} - mean weighted F1 (%) over the three splits per perturbation level and drop 0%->40%.", st["caption"]))
    mr = R["rob_sum"].sort_values("F1 drop 0%->40%")
    S.append(P(f"Based on these results, <b>{mr.iloc[0]['Model']}</b> shows the smallest mean F1 drop from clean data "
               f"to 40% perturbation ({mr.iloc[0]['F1 drop 0%->40%']:.2f} percentage points) and "
               f"<b>{mr.iloc[-1]['Model']}</b> the largest ({mr.iloc[-1]['F1 drop 0%->40%']:.2f}).", st["body"]))
    S.append(fig_block(fg["rob_acc_all"], f"Figure: {label} - accuracy vs perturbation, all splits (line style = split).", st, 14 * cm, 9 * cm))
    S.append(fig_block(fg["rob_f1_all"], f"Figure: {label} - weighted F1 vs perturbation, all splits (line style = split).", st, 14 * cm, 9 * cm))

    # 9 Confusion matrices
    S.append(PageBreak())
    S.append(P(escape(f"9. {label} - Confusion Matrices"), st["h2"]))
    S.append(P(f"{PRIMARY_SPLIT} split test set. Rows = actual class, columns = predicted class, values = number of "
               "test samples.", st["body"]))
    for m in MODELS:
        S.append(fig_block(fg["cm"][m], f"Figure: {label} - confusion matrix, {m} ({PRIMARY_SPLIT} split).", st, 10.5 * cm, 8.8 * cm))

    # 10 Consistency
    S.append(PageBreak())
    S.append(P(escape(f"10. {label} - Model Consistency"), st["h2"]))
    S.append(P("Mean and sample standard deviation (ddof=1) over the three train/test splits (%).", st["body"]))
    sm = R["summary"][["Model", "Mean Accuracy", "Std Accuracy", "Mean Precision", "Std Precision",
                       "Mean Recall", "Std Recall", "Mean F1", "Std F1"]]
    S.append(df_table(sm, st, col_widths=[1.6 * cm] + [1.95 * cm] * 8, decimals=3))
    S.append(P(f"Table: {label} - mean and standard deviation of the metrics across splits.", st["caption"]))
    S.append(fig_block(fg["mean_acc"], f"Figure: {label} - mean accuracy (%) by model.", st, 13 * cm, 8 * cm))
    S.append(fig_block(fg["mean_f1"], f"Figure: {label} - mean weighted F1 (%) by model.", st, 13 * cm, 8 * cm))
    S.append(fig_block(fg["acc_across"], f"Figure: {label} - accuracy (%) across splits.", st, 14 * cm, 8.5 * cm))
    S.append(fig_block(fg["f1_across"], f"Figure: {label} - weighted F1 (%) across splits.", st, 14 * cm, 8.5 * cm))

    # 11 Best model
    S.append(P(escape(f"11. {label} - Best Model"), st["h2"]))
    bt = pd.DataFrame([{"Criterion": f"Best model - {l}", "Model": R["best_by_split"][l][0],
                        "Weighted F1 (%)": R["best_by_split"][l][1], "Accuracy (%)": R["best_by_split"][l][2]}
                       for l in SPLIT_LABELS])
    S.append(df_table(bt, st, col_widths=[4.5 * cm, 4 * cm, 3.8 * cm, 3.8 * cm]))
    S.append(P(f"Table: {label} - best model per split (highest weighted F1; ties broken by accuracy; exact ties reported).", st["caption"]))
    cs = R["cons"][["Model", "Mean F1", "Std F1", "Consistency Score (Mean F1 - Std F1)"]]
    S.append(df_table(cs, st, col_widths=[3 * cm, 3.5 * cm, 3.5 * cm, 6 * cm], decimals=3))
    S.append(P(f"Table: {label} - models ranked by consistency score (mean F1 minus std of F1).", st["caption"]))
    S.append(P(f"<b>Overall most consistent model on {label}: {escape(R['most_consistent'])}</b>.", st["body"]))

    # 12 Conclusion
    S.append(P(escape(f"12. {label} - Conclusion"), st["h2"]))
    for c in conclusion_points(R):
        S.append(P(escape(c), st["body"]))


def build_comparison_section(S, results, st, all_perf, all_rob, all_summary, cmp_figs):
    S.append(PageBreak())
    S.append(P("Final Cross-Dataset Comparison", st["h1"]))
    S.append(P("This section only compares the results of the independent analyses. No data were merged: each "
               "dataset was split, preprocessed, trained and evaluated on its own, and the values below are means "
               "over the three train/test splits unless stated otherwise.", st["body"]))

    # dataset overview
    ov = pd.DataFrame([{"Dataset": r["label"], "File": r["file"], "Sheet": str(r["sheet"]) if r["sheet"] else "CSV",
                        "Rows used": r["n_samples"],
                        "Class counts": ", ".join(f"{c}: {n}" for c, n in zip(r["class_table"]["Class"],
                                                                              r["class_table"]["Number of Samples"]))}
                       for r in results])
    S.append(df_table(ov, st, col_widths=[1.9 * cm, 5 * cm, 3 * cm, 1.8 * cm, 5 * cm]))
    S.append(P("Table: overview of the analysed datasets.", st["caption"]))

    def mean_table(metric_cols):
        rows = []
        for r in results:
            for m in MODELS:
                d = all_perf[(all_perf.Dataset == r["label"]) & (all_perf.Model == m)]
                row = {"Dataset": r["label"], "Model": m}
                for c in metric_cols:
                    row[f"Mean {c}"] = d[c].mean()
                rows.append(row)
        return pd.DataFrame(rows)

    sections = [("1. Overall Accuracy Comparison", "Accuracy", "overall_accuracy_comparison"),
                ("2. Overall F1 Comparison", "F1", "overall_f1_comparison"),
                ("3. Overall Precision Comparison", "Precision", "overall_precision_comparison"),
                ("4. Overall Recall Comparison", "Recall", "overall_recall_comparison")]
    for title, metric, key in sections:
        S.append(P(title, st["h2"]))
        mt = mean_table([metric])
        S.append(df_table(mt, st, col_widths=[3 * cm, 3 * cm, 4 * cm]))
        S.append(P(f"Table: mean {metric} (%) of every model on every dataset (mean of the three splits).", st["caption"]))
        top_row = mt.sort_values(f"Mean {metric}", ascending=False).iloc[0]
        S.append(P(f"The highest mean {metric.lower()} was {top_row[f'Mean {metric}']:.2f}% "
                   f"({top_row['Model']} on {top_row['Dataset']}).", st["body"]))
        S.append(fig_block(cmp_figs[key], f"Figure: overall {metric.lower()} comparison across the four datasets.", st, 16 * cm, 9.5 * cm))

    # robustness
    S.append(PageBreak())
    S.append(P("5. Robustness Comparison", st["h2"]))
    rrows = []
    for r in results:
        for _, row in r["rob_sum"].iterrows():
            rrows.append({"Dataset": r["label"], **row.to_dict()})
    rcmp = pd.DataFrame(rrows)
    S.append(df_table(rcmp, st, col_widths=[1.9 * cm, 1.6 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm]))
    S.append(P("Table: mean weighted F1 (%) of the fixed models under 0% (clean), 20%, 30% and 40% test-feature "
               "perturbation, and the drop from 0% to 40% (mean over splits).", st["caption"]))
    S.append(fig_block(cmp_figs["overall_robustness_comparison"],
                       "Figure: robustness comparison - mean weighted F1 vs perturbation level per dataset.", st, 16 * cm, 14 * cm))

    # consistency
    S.append(PageBreak())
    S.append(P("6. Model Consistency Comparison", st["h2"]))
    ccmp = all_summary[["Dataset", "Model", "Mean F1", "Std F1", "Consistency Score (Mean F1 - Std F1)"]]
    S.append(df_table(ccmp, st, col_widths=[2.6 * cm, 2.6 * cm, 3 * cm, 3 * cm, 5 * cm], decimals=3))
    S.append(P("Table: mean and standard deviation of weighted F1 across the three splits, per dataset and model.", st["caption"]))
    S.append(fig_block(cmp_figs["overall_consistency_comparison"],
                       "Figure: standard deviation of weighted F1 across splits (lower = more consistent).", st, 16 * cm, 9.5 * cm))

    # best models
    S.append(P("7. Best Model Across Each Dataset", st["h2"]))
    brow = []
    for r in results:
        mr = r["rob_sum"].sort_values("F1 drop 0%->40%").iloc[0]["Model"]
        brow.append({"Dataset": r["label"], "Best 80/20": r["best_by_split"]["80/20"][0],
                     "Best 70/30": r["best_by_split"]["70/30"][0], "Best 60/40": r["best_by_split"]["60/40"][0],
                     "Most consistent": r["most_consistent"], "Most robust (smallest F1 drop)": mr})
    bdf = pd.DataFrame(brow)
    S.append(df_table(bdf, st, col_widths=[1.9 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3.2 * cm, 3.6 * cm]))
    S.append(P("Table: best model per dataset and split (highest weighted F1), most consistent model and most robust model.", st["caption"]))

    # findings
    S.append(P("8. Overall Findings", st["h2"]))
    cons_models = {r["label"]: base_model(r["most_consistent"]) for r in results}
    if len(set(cons_models.values())) == 1:
        f0 = f"{list(cons_models.values())[0]} is the most consistent model on every dataset."
    else:
        f0 = "The most consistent model differs between datasets: " + ", ".join(
            f"{r['label']}: {r['most_consistent']}" for r in results) + "."
    model_mean_f1 = all_perf.groupby("Model")["F1"].mean().reindex(MODELS)
    ds_mean_f1 = all_perf.groupby("Dataset")["F1"].mean()
    ds_mean_f1 = ds_mean_f1.reindex([r["label"] for r in results])
    drop_by_model = rcmp.groupby("Model")["F1 drop 0%->40%"].mean().reindex(MODELS)
    sep = [r["label"] for r in results if r["classes_separable"]]
    findings = [
        f0,
        f"Averaged over all datasets and splits, the highest mean weighted F1 belongs to {model_mean_f1.idxmax()} "
        f"({model_mean_f1.max():.2f}%) and the lowest to {model_mean_f1.idxmin()} ({model_mean_f1.min():.2f}%).",
        f"Averaged over all models, {ds_mean_f1.idxmax()} gave the highest mean weighted F1 ({ds_mean_f1.max():.2f}%) "
        f"and {ds_mean_f1.idxmin()} the lowest ({ds_mean_f1.min():.2f}%); the spread between datasets is "
        f"{ds_mean_f1.max() - ds_mean_f1.min():.2f} percentage points.",
        f"Under 40% test-feature perturbation, the mean weighted F1 drop (over datasets, splits) was smallest for "
        f"{drop_by_model.idxmin()} ({drop_by_model.min():.2f} percentage points) and largest for "
        f"{drop_by_model.idxmax()} ({drop_by_model.max():.2f} percentage points).",
    ]
    if sep:
        findings.append("On " + ", ".join(sep) + (" the" if len(sep) == 1 else " the") +
                        " AVG_NOISE_LVL ranges of the risk classes do not overlap, so scores there largely reflect how "
                        "well a model recovers the noise-level thresholds.")
    findings.append("Class counts differ between the datasets (see the overview table), so absolute scores are "
                    "comparable only in the sense that each dataset was evaluated with the same protocol.")
    for f in findings:
        S.append(P(escape(f), st["body"]))

    # final conclusion
    S.append(P("9. Final Conclusion", st["h2"]))
    fin = [
        "The same protocol (three stratified splits, training-only preprocessing, four models, three perturbation "
        "levels) was applied independently to each dataset. Best models per dataset: " +
        "; ".join(f"{r['label']} - {r['most_consistent']}" for r in results) + ".",
        f"Across datasets {model_mean_f1.idxmax()} had the highest mean weighted F1 ({model_mean_f1.max():.2f}%), while "
        f"{drop_by_model.idxmin()} was the least affected by 40% test-feature noise.",
        "All numbers in this report were computed from the supplied datasets by this script and stored in the CSV "
        "files in the results folder; conclusions are limited to these datasets and this experimental setup.",
    ]
    for f in fin:
        S.append(P(escape(f), st["body"]))


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cmp_dir = os.path.join(RESULTS_DIR, "overall_comparison")
    os.makedirs(cmp_dir, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    banner("DATASET DETECTION")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cands = find_datasets(arg)
    print("\nDatasets selected (analysed separately, never merged):")
    for i, c in enumerate(cands, start=1):
        print(f"  Dataset {i}: {os.path.basename(c['path'])}"
              f"{' | sheet: ' + str(c['sheet']) if c['sheet'] else ''} | rows: {c['rows']}")

    results = []
    for i, c in enumerate(cands, start=1):
        results.append(analyze_dataset(i, c))      # completely independent analysis

    # ---------------- combined summary CSVs (summaries only; no training here) ----------------
    banner("COMBINED SUMMARIES AND CROSS-DATASET GRAPHS")
    perf_parts, rob_parts, sum_parts = [], [], []
    for r in results:
        p = r["perf"].copy()
        p.insert(0, "Dataset", r["label"])
        p["File"] = r["file"]
        perf_parts.append(p)
        q = r["rob"].copy()
        q.insert(0, "Dataset", r["label"])
        q["File"] = r["file"]
        rob_parts.append(q)
        s_ = r["summary"].copy()
        s_.insert(0, "Dataset", r["label"])
        s_["File"] = r["file"]
        sum_parts.append(s_)
    all_perf = pd.concat(perf_parts, ignore_index=True)
    all_rob = pd.concat(rob_parts, ignore_index=True)
    all_summary = pd.concat(sum_parts, ignore_index=True)
    all_perf.to_csv(os.path.join(cmp_dir, "ALL_DATASETS_model_performance.csv"), index=False)
    all_rob.to_csv(os.path.join(cmp_dir, "ALL_DATASETS_robustness_results.csv"), index=False)
    all_summary.to_csv(os.path.join(cmp_dir, "ALL_DATASETS_model_summary.csv"), index=False)

    cmp_figs = {}
    for metric, key in [("Accuracy", "overall_accuracy_comparison"), ("F1", "overall_f1_comparison"),
                        ("Precision", "overall_precision_comparison"), ("Recall", "overall_recall_comparison")]:
        cmp_figs[key] = os.path.join(cmp_dir, f"{key}.png")
        plot_overall_metric(results, all_perf, metric, cmp_figs[key])
    cmp_figs["overall_robustness_comparison"] = os.path.join(cmp_dir, "overall_robustness_comparison.png")
    plot_overall_robustness(results, cmp_figs["overall_robustness_comparison"])
    cmp_figs["overall_consistency_comparison"] = os.path.join(cmp_dir, "overall_consistency_comparison.png")
    plot_overall_consistency(results, all_summary, cmp_figs["overall_consistency_comparison"])
    print("Cross-dataset graphs and CSV files saved in:", os.path.abspath(cmp_dir))

    # ---------------- PDF ----------------
    banner("BUILDING THE PDF REPORT")
    st = make_styles()
    S = []
    S.append(Spacer(1, 2.2 * cm))
    S.append(P("ROAD NOISE RISK LEVEL PREDICTION<br/>USING MACHINE LEARNING", st["title"]))
    S.append(Spacer(1, 0.4 * cm))
    S.append(P("Complete Comparative Analysis of Four Datasets", st["sub"]))
    S.append(Spacer(1, 1 * cm))
    ds_rows = [[Paragraph("<b>Dataset</b>", st["cell"]), Paragraph("<b>File</b>", st["cell"]),
                Paragraph("<b>Sheet</b>", st["cell"]), Paragraph("<b>Rows used</b>", st["cell"]),
                Paragraph("<b>Classes</b>", st["cell"])]]
    for r in results:
        ds_rows.append([Paragraph(r["label"], st["cell"]), Paragraph(escape(r["file"]), st["cell"]),
                        Paragraph(escape(str(r["sheet"]) if r["sheet"] else "CSV"), st["cell"]),
                        Paragraph(str(r["n_samples"]), st["cell"]),
                        Paragraph(escape(", ".join(r["class_names"])), st["cell"])])
    t = Table(ds_rows, colWidths=[2.2 * cm, 6 * cm, 3.3 * cm, 2 * cm, 3.2 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S.append(t)
    S.append(Spacer(1, 0.8 * cm))
    cfg = [["Target variable", TARGET],
           ["Features", ", ".join(REQUIRED_FEATURES)],
           ["Models", "LR, LMR, XGB, RF"],
           ["Train/test splits", ", ".join(SPLIT_LABELS) + " (stratified)"],
           ["Robustness levels", ", ".join(f"{p}%" for p in PERTURBATIONS) + " Gaussian perturbation of test features"],
           ["Random state", str(RANDOM_STATE)]]
    t2 = Table([[Paragraph(f"<b>{escape(a)}</b>", st["cell"]), Paragraph(escape(b), st["cell"])] for a, b in cfg],
               colWidths=[4.5 * cm, 12.2 * cm], hAlign="CENTER")
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f8")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S.append(t2)
    S.append(PageBreak())

    S.append(P("Table of Contents", st["tocT"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC0", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=7, leftIndent=0),
        ParagraphStyle("TOC1", fontName="Helvetica", fontSize=9, leading=12, leftIndent=16),
    ]
    S.append(toc)

    for r in results:
        build_dataset_section(S, r, st)
    build_comparison_section(S, results, st, all_perf, all_rob, all_summary, cmp_figs)

    doc = ReportDoc(PDF_PATH, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                    bottomMargin=2 * cm, title="Road Noise Risk Level Prediction - Four-Dataset Analysis",
                    author="Road Noise ML Experiment")
    doc.multiBuild(S, onFirstPage=page_number, onLaterPages=page_number)

    # ---------------- final terminal summary ----------------
    banner("FOUR-DATASET ML ANALYSIS COMPLETED")
    for r in results:
        print(f"\n{r['label']} ({r['file']}):")
        for lab in SPLIT_LABELS:
            print(f"Best Model - {lab}: {r['best_by_split'][lab][0]}")
        print(f"Most Consistent Model: {r['most_consistent']}")
    banner("FINAL CROSS-DATASET COMPARISON")
    for r in results:
        print(f"{r['label']} -> Best Model: {r['most_consistent']}")
    print("\n" + "=" * 70)
    print("\nCOMPLETE PDF:\n")
    print(PDF_PATH.replace(os.sep, "/"))
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
