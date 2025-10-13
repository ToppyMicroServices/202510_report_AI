import pandas as pd
import numpy as np
from typing import List, Dict
import argparse
import os
import matplotlib.pyplot as plt

# =============================
# Configurable metric aliases
# =============================
METRIC_ALIASES: Dict[str, List[str]] = {
    "ece_w": ["ece_w", "ece_equal_width", "ece_width"],
    "ece_f": ["ece_f", "ece_equal_freq", "ece_frequency"],
    "brier": ["brier", "brier_score"],
    "logloss": ["logloss", "log_loss", "nll"],
    "halluc_rate": ["halluc_rate", "hallucination_rate", "hallucinations", "halluc", "overconfident"]
}

REQUIRED_KEYS = ["model", "prompt", "condition", "item_id"]
CONDITION_BASE = "C0"
CONDITIONS = ["C0", "C1", "C2", "C3"]

# -----------------------------
# CLI options (schema- and key-control)
# -----------------------------
parser = argparse.ArgumentParser(description="Paired condition deltas (C0 vs C1/C2/C3) with schema-agnostic joins")
parser.add_argument("--base", default=CONDITION_BASE, help="Baseline condition label (default: C0)")
parser.add_argument("--conds", default="C1,C2,C3", help="Comma-separated comparison conditions (default: C1,C2,C3)")
parser.add_argument("--ignore-prompt", action="store_true", default=True, help="Ignore prompt when pairing (default: enabled, useful if prompt text changes across conditions)")
parser.add_argument("--force-item-key", choices=["auto","item_id","domain+qid","qid"], default="domain+qid",
                    help="Override how to build item key (default: domain+qid)")
parser.add_argument("--model-col", default=None, help="Explicit model column name if not 'model' or 'model_name'")
parser.add_argument("--prompt-col", default=None, help="Explicit prompt column name if not 'prompt'/'task' etc.")
parser.add_argument("--cond-col", default=None, help="Explicit condition column name if not 'condition'/'cond'")
parser.add_argument("--itemid-col", default=None, help="Explicit item id column (e.g., 'qid')")
args = parser.parse_args()

CONDITION_BASE = args.base
CONDITIONS = [c.strip() for c in args.conds.split(',') if c.strip()]
IGNORE_PROMPT = bool(args.ignore_prompt)
FORCE_ITEM_KEY = args.force_item_key
EXPLICIT_COL_OVERRIDES = {k[:-4] + ("" if k == "itemid-col" else "_col"): v for k, v in {
    "model-col": args.model_col,
    "prompt-col": args.prompt_col,
    "cond-col": args.cond_col,
    "itemid-col": args.itemid_col,
}.items() if v is not None}

# -----------------------------
# Utilities
# -----------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve canonical column names (lowercased) from potential aliases."""
    resolved = {}
    cols = set(df.columns)

    # explicit overrides from CLI
    overrides = {
        "model": EXPLICIT_COL_OVERRIDES.get("model_col"),
        "prompt": EXPLICIT_COL_OVERRIDES.get("prompt_col"),
        "condition": EXPLICIT_COL_OVERRIDES.get("cond_col"),
        "item_id": EXPLICIT_COL_OVERRIDES.get("itemid_col"),
    }

    # required keys
    for k in REQUIRED_KEYS:
        if overrides.get(k) and overrides[k] in cols:
            resolved[k] = overrides[k]
            continue
        if k in cols:
            resolved[k] = k
        else:
            # try some common variants
            variants = {
                "model": ["model", "model_name"],
                "prompt": ["prompt", "prompt_name", "task"],
                "condition": ["condition", "cond"],
                "item_id": ["item_id", "qid", "example_id", "sample_id"],
            }[k]
            found = next((v for v in variants if v in cols), None)
            if not found:
                raise KeyError(f"Missing required column '{k}'. Available columns: {sorted(cols)}")
            resolved[k] = found

    # metrics
    for canon, aliases in METRIC_ALIASES.items():
        found = next((a for a in aliases if a in cols), None)
        if found is None:
            # metric optional: we can skip if not present
            continue
        resolved[canon] = found

    return resolved


def _metrics_present(resolved: Dict[str, str]) -> List[str]:
    mets = [m for m in METRIC_ALIASES.keys() if m in resolved]
    if not mets:
        raise KeyError("No known metric columns found. Expected one of: " + \
                       ", ".join(["/".join(v) for v in METRIC_ALIASES.values()]))
    return mets


# =============================
# Pairing helpers
# =============================
def _build_item_key(df: pd.DataFrame, resolved: Dict[str, str]) -> pd.Series:
    cols = set(df.columns)
    if FORCE_ITEM_KEY == "item_id":
        if resolved.get("item_id") in cols:
            return df[resolved["item_id"]].astype(str)
    elif FORCE_ITEM_KEY == "domain+qid":
        if "domain" in cols and (resolved.get("item_id") == "qid" or "qid" in cols):
            qcol = resolved.get("item_id") if resolved.get("item_id") in cols else "qid"
            return df["domain"].astype(str) + "::" + df[qcol].astype(str)
    elif FORCE_ITEM_KEY == "qid":
        if "qid" in cols:
            return df["qid"].astype(str)
    # Prefer explicit item_id; fall back to (domain,qid) or qid if available
    if resolved.get("item_id") in cols:
        return df[resolved["item_id"]].astype(str)
    if "domain" in cols and (resolved.get("item_id") == "qid" or "qid" in cols):
        qcol = resolved.get("item_id") if resolved.get("item_id") in cols else "qid"
        return df["domain"].astype(str) + "::" + df[qcol].astype(str)
    # last resort: use qid if present
    if "qid" in cols:
        return df["qid"].astype(str)
    raise KeyError("Cannot construct item key; need one of item_id/qid or (domain,qid)")


def _try_pair_merge(base: pd.DataFrame, comp: pd.DataFrame, resolved: Dict[str, str], cond: str):
    base = base.copy(); comp = comp.copy()
    base["__item_key__"] = _build_item_key(base, resolved)
    comp["__item_key__"] = _build_item_key(comp, resolved)

    # Candidate join key sets in order of strictness
    key_sets = []
    if not IGNORE_PROMPT:
        key_sets.append([resolved["model"], resolved["prompt"], "__item_key__"])
    key_sets.append([resolved["model"], "__item_key__"])
    key_sets.append(["__item_key__"])

    for keys in key_sets:
        # ensure keys exist in both frames
        ok = all(k in base.columns and k in comp.columns for k in keys)
        if not ok:
            continue
        merged = base.merge(
            comp,
            on=keys,
            suffixes=("_C0", f"_tmp")
        )
        if not merged.empty:
            # rename metric suffix for comp condition after successful merge
            for col in list(merged.columns):
                if col.endswith("_tmp"):
                    merged.rename(columns={col: col[:-4] + f"_{cond}"}, inplace=True)
            return merged
    # Fall-through: empty
    return pd.DataFrame()


def bootstrap_ci(series: pd.Series, n_boot: int = 2000, alpha: float = 0.05):
    s = series.dropna().to_numpy()
    if s.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0)
    means = np.empty(n_boot, dtype=float)
    n = s.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = s[idx].mean()
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return (lo, hi)


# -----------------------------
# Main
# -----------------------------

# -----------------------------
# Reporting helpers (LaTeX table & forest plot)
# -----------------------------

def _ensure_dirs():
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def _select_metric_for_forest(summary_df: pd.DataFrame) -> str:
    preferred = ["ece_f", "ece_w", "brier", "logloss"]
    avail = list(dict.fromkeys(summary_df["metric"].tolist()))
    for m in preferred:
        if m in avail:
            return m
    return avail[0]


def _make_latex_table(summary_df: pd.DataFrame, outpath: str = "tables/tab_paired_deltas.tex"):
    # choose columns to show
    metrics_order = [m for m in ["ece_w", "ece_f", "brier", "logloss", "halluc_rate"] if m in summary_df["metric"].unique()]
    df = summary_df.copy()
    df = df[df["metric"].isin(metrics_order)]

    # format mean [lo, hi]
    df["ci"] = df.apply(lambda r: f"{r['delta_mean']:.3f} [{r['delta_lo']:.3f}, {r['delta_hi']:.3f}]", axis=1)
    # Keep consistent ordering
    df["metric"] = pd.Categorical(df["metric"], categories=metrics_order, ordered=True)

    # pivot to Model | Prompt | Condition x metrics (explicitly set observed=False)
    piv = df.pivot_table(index=["model", "prompt", "condition"], columns="metric", values="ci", aggfunc="first", observed=False)
    piv = piv.sort_index()

    # --- Build LaTeX manually (no Styler/Jinja2 dependency) ---
    cols = ["model", "prompt", "condition"] + list(metrics_order)
    header = " \\toprule\n" + " & ".join([c.replace("_", "\\_") for c in cols]) + " \\\\ \\midrule\n"
    lines = []
    for (mod, prm, cond), row in piv.iterrows():
        cells = [str(mod), str(prm), str(cond)] + [str(row.get(m, "")) for m in metrics_order]
        lines.append(" & ".join(cells) + " \\\\")
    body = "\n".join(lines) + "\n\\bottomrule\n"

    latex = (
        "% Auto-generated by analyze_condition_deltas.py\n"
        "% Mean deltas (cond - C0) with BCa 95% CI; negative is improvement.\n"
        "\\begin{tabular}{" + ("l" * len(cols)) + "}\n" + header + body + "\\end{tabular}\n"
    )

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(latex)


def _make_forest_plot(summary_df: pd.DataFrame, basepath: str = "figures/fig_delta_ece_forest"):
    # pick a metric to visualize
    metric = _select_metric_for_forest(summary_df)
    df = summary_df[summary_df["metric"] == metric].copy()
    if df.empty:
        return

    # build y labels
    df = df.sort_values(["model", "condition", "prompt"]).reset_index(drop=True)
    labels = (df["model"].astype(str) + " | " + df["condition"].astype(str) +
              (" | " + df["prompt"].astype(str) if (df["prompt"].nunique() > 1 and df["prompt"].iloc[0] != "(no-prompt)") else ""))

    y = np.arange(len(df))
    x = df["delta_mean"].to_numpy()
    xlo = x - df["delta_lo"].to_numpy()
    xhi = df["delta_hi"].to_numpy() - x

    plt.figure()
    plt.errorbar(x, y, xerr=[xlo, xhi], fmt='o')
    plt.axvline(0, linestyle='--')
    plt.yticks(y, labels)
    plt.xlabel(f"Δ{metric} (cond − C0)  (negative = improvement)")
    plt.tight_layout()
    pdf_path = basepath + ".pdf"
    png_path = basepath + ".png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=600)
    plt.close()
    return pdf_path, png_path

df = pd.read_csv("results_llm_experiment.csv")

# normalize names
df = _normalize_columns(df)

# If 'model' is absent, assume a single-model run and synthesize it
if "model" not in df.columns and "model_name" not in df.columns:
    df["model"] = "(unspecified)"

# If per-item metrics are not present but we have confidence/correct,
# derive brier and logloss per row so that paired deltas can be computed.
if "brier" not in df.columns or "logloss" not in df.columns:
    if "confidence" in df.columns and "correct" in df.columns:
        p = df["confidence"].astype(float).clip(1e-6, 1-1e-6)
        y = df["correct"].astype(float)
        if "brier" not in df.columns:
            df["brier"] = (p - y) ** 2
        if "logloss" not in df.columns:
            df["logloss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))

resolved = _resolve_columns(df)
metrics = _metrics_present(resolved)

# filter to known conditions actually present
conds_available = sorted(set(df[resolved["condition"]].unique()))
conds_to_use = [c for c in CONDITIONS if c in conds_available and c != CONDITION_BASE]
if not conds_to_use:
    raise ValueError(f"No comparison conditions found. Present conditions: {conds_available}")

print("[schema] columns:", sorted(df.columns))
print("[schema] conditions present:", sorted(df[resolved["condition"]].unique()))
print("[schema] sample rows:")
print(df.head(3).to_string(index=False))

# prepare base
base = df[df[resolved["condition"]] == CONDITION_BASE]

# join & deltas
pair_parts = []
for cond in conds_to_use:
    comp = df[df[resolved["condition"]] == cond]
    merged = _try_pair_merge(base, comp, resolved, cond)
    if merged.empty:
        continue
    # capture actual condition label from comp side
    if "cond" not in merged.columns:
        merged["cond"] = cond
    # compute deltas
    for m in metrics:
        # find suffixed columns
        cand_c0 = [c for c in merged.columns if c.startswith(resolved[m] + "_") and c.endswith("_C0")]
        cand_cx = [c for c in merged.columns if c.startswith(resolved[m] + "_") and c.endswith(f"_{cond}")]
        if not cand_c0 or not cand_cx:
            continue
        m0 = cand_c0[0]
        mc = cand_cx[0]
        merged[f"delta_{m}"] = merged[mc] - merged[m0]
        merged[f"rel_{m}"] = np.where(merged[m0] != 0, merged[f"delta_{m}"] / merged[m0], np.nan)
    pair_parts.append(merged)

if not pair_parts:
    # diagnostics to help user inspect mismatches
    diag = {
        "conditions_present": conds_available,
        "base_count": int(base.shape[0]),
    }
    print("[diagnostics]", diag)
    raise ValueError("No paired items after merge. Try ensuring consistent qid/item_id across conditions, or include a 'domain' column to pair by (domain,qid). Also verify that 'prompt' strings do not change across conditions.")

out = pd.concat(pair_parts, ignore_index=True)

# long-format for deltas
# Decide grouping keys dynamically (prompt may be suffixed or ignored)
rows = []

# choose prompt grouping column
prompt_key = None
if not IGNORE_PROMPT:
    for cand in [resolved["prompt"], f"{resolved['prompt']}_C0"]:
        if cand in out.columns:
            prompt_key = cand
            break

group_keys = [resolved["model"], "cond"]
if prompt_key is not None:
    group_keys.insert(1, prompt_key)

for keys, g in out.groupby(group_keys):
    # unpack keys
    if prompt_key is None:
        mod, cond = keys if isinstance(keys, tuple) else (keys,)
        prm_val = "(no-prompt)"
    else:
        mod, prm_val, cond = keys
    for m in metrics:
        if f"delta_{m}" not in g.columns:
            continue
        mean = g[f"delta_{m}"].mean()
        lo, hi = bootstrap_ci(g[f"delta_{m}"])
        rows.append({
            "model": mod,
            "prompt": prm_val,
            "condition": cond,
            "metric": m,
            "delta_mean": mean,
            "delta_lo": lo,
            "delta_hi": hi,
            "n_pairs": int(g.shape[0])
        })


summary = pd.DataFrame(rows).sort_values(["metric", "model", "prompt", "condition"]) 
summary.to_csv("condition_deltas_summary.csv", index=False)

# also save the detailed paired dataset for auditability
# choose prompt column (if any)
keep_prompt_col = None
for cand in [resolved["prompt"], f"{resolved['prompt']}_C0"]:
    if cand in out.columns:
        keep_prompt_col = cand
        break

# choose item identifier column robustly
item_col_candidates = [
    "__item_key__",
    resolved.get("item_id", "item_id"),
    f"{resolved.get('item_id', 'item_id')}_C0",
    "qid",
    "item_id",
]
keep_item_col = next((c for c in item_col_candidates if c in out.columns), None)

base_keep = [resolved["model"], "cond"]
if keep_prompt_col is not None:
    base_keep.insert(1, keep_prompt_col)
if keep_item_col is not None:
    base_keep.insert(1, keep_item_col)

cols_keep = base_keep + [f"delta_{m}" for m in metrics] + [f"rel_{m}" for m in metrics]

# reduce to existing columns only to be safe
cols_keep = [c for c in cols_keep if c in out.columns]

out[cols_keep].to_csv("condition_deltas_long.csv", index=False)

print("Saved → condition_deltas_summary.csv, condition_deltas_long.csv")

# ---- Auto-generate LaTeX table and figures by default ----
_ensure_dirs()
# table (robust, no Jinja2)
try:
    _make_latex_table(summary)
    print("Also wrote table: tables/tab_paired_deltas.tex")
except Exception as e:
    print("[warn] Table build failed:", e)
# figure (always attempt even if table failed)
try:
    paths = _make_forest_plot(summary)
    if paths:
        print("Also wrote figures:", paths[0] + ",", paths[1])
    else:
        print("[warn] No suitable metric for figure; skipped.")
except Exception as e:
    print("[warn] Figure build failed:", e)
