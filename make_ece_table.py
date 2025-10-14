#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ece_table.py
- Aggregate metrics by condition × domain (mean, count, 95% CI) from
  results_llm_experiment.csv and export both CSV and LaTeX (booktabs).

Dependencies: pandas, numpy
Defaults:
  Input:  results_llm_experiment.csv
  Output: summary_by_condition_domain.csv, Tables/tab_ece_condition_domain.tex
  Percent display: accuracy, overconfident

Example:
  python make_ece_table.py
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

def detect_column(lower_map: Dict[str, str], *candidates: str) -> Optional[str]:
    for cand in candidates:
        col = lower_map.get(cand.lower().strip())
        if col: return col
    return None

def format_float(x: float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def main():
    ap = argparse.ArgumentParser(description="Aggregate LLM experiment metrics by condition × domain.")
    ap.add_argument(
        "csv",
        nargs="?",
        default="results_llm_experiment.csv",
        help="Input CSV (default: results_llm_experiment.csv)",
    )
    ap.add_argument("--out-csv", default="summary_by_condition_domain.csv", help="Output aggregated CSV path")
    ap.add_argument("--out-tex", default="Tables/tab_ece_condition_domain.tex", help="Output LaTeX table path")
    # Display options
    ap.add_argument("--digits", type=int, default=3, help="Decimal places for LaTeX values")
    ap.add_argument(
        "--as-perc",
        nargs="*",
        default=["accuracy", "overconfident"],
        help="Column names to display as percent (e.g., accuracy, overconfident)",
    )
    ap.add_argument("--title", default=None, help="LaTeX title (unused by default; table is a bare tabular)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # Column map (lowercase → original)
    lower_map = {c.lower().strip(): c for c in df.columns}
    # Required grouping columns
    condition_col = detect_column(lower_map, "condition")
    domain_col    = detect_column(lower_map, "domain")
    if not condition_col:
        print("Required column not found: condition", file=sys.stderr); sys.exit(1)
    if not domain_col:
        print("Required column not found: domain", file=sys.stderr); sys.exit(1)

    # Metric columns present in results_llm_experiment.csv
    # correct → accuracy; confidence; overconfident
    metric_map = []
    if "correct" in df.columns:
        df["accuracy"] = pd.to_numeric(df["correct"], errors="coerce")
        metric_map.append(("accuracy", "accuracy"))
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        metric_map.append(("confidence", "confidence"))
    if "overconfident" in df.columns:
        df["overconfident"] = pd.to_numeric(df["overconfident"], errors="coerce")
        metric_map.append(("overconfident", "overconfident"))
    if not metric_map:
        print("No expected metric columns found (correct/confidence/overconfident).", file=sys.stderr)
        sys.exit(1)

    # Grouping keys
    group_keys = [condition_col, domain_col]
    g = df.groupby(group_keys, dropna=False)

    # Compute mean, std, n, and 95% CI (normal approximation)
    rows = []
    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {condition_col: keys[0], domain_col: keys[1]}
        n = len(sub)
        rec["N"] = n
        for name, col in metric_map:
            x = sub[col].dropna()
            if len(x) == 0:
                rec[f"{name}_mean"] = np.nan
                rec[f"{name}_ci_low"] = np.nan
                rec[f"{name}_ci_high"] = np.nan
                continue
            mu = float(x.mean())
            sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            se = sd / math.sqrt(len(x)) if len(x) > 1 else 0.0
            z = 1.96
            rec[f"{name}_mean"]   = mu
            rec[f"{name}_ci_low"] = mu - z*se
            rec[f"{name}_ci_high"]= mu + z*se
        rows.append(rec)

    agg = pd.DataFrame(rows)

    # Convert selected columns to percent (*_mean / *_ci_low / *_ci_high)
    as_perc = set(s.lower() for s in args.as_perc)
    for name, _col in metric_map:
        if name.lower() in as_perc:
            for suf in ("mean", "ci_low", "ci_high"):
                c = f"{name}_{suf}"
                if c in agg.columns:
                    agg[c] = agg[c] * 100.0

    # Write CSV
    out_csv_path = Path(args.out_csv)
    if out_csv_path.parent and not out_csv_path.parent.exists():
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv_path, index=False)

    # LaTeX table body (show means only, with N)
    show_names = [name for name, _ in metric_map]  # display order
    header = ["Condition", "Domain"] + [n.replace("_"," ").title() for n in show_names] + ["N"]

    # LaTeX alignment spec
    align = "ll" + "r"*len(show_names) + "r"

    # Build LaTeX rows
    lines = []
    lines.append(r"\begin{tabular}{%s}" % align)
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Sort by Condition → Domain
    sort_cols = [condition_col, domain_col]
    agg_sort = agg.sort_values(sort_cols)

    def pick_mean(row, name):
        val = row.get(f"{name}_mean", np.nan)
        return format_float(val, args.digits)

    for _, row in agg_sort.iterrows():
        vals = [str(row[condition_col]), str(row[domain_col])]
        for name in show_names:
            vals.append(pick_mean(row, name))
        vals.append(str(int(row["N"])))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # LaTeX output
    out_tex = "\n".join(lines)
    # Create parent directory if missing
    out_tex_path = Path(args.out_tex)
    if out_tex_path.parent and not out_tex_path.parent.exists():
        out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text(out_tex, encoding="utf-8")

    # Progress
    sys.stderr.write(f"[ok] wrote CSV: {args.out_csv}\n")
    sys.stderr.write(f"[ok] wrote LaTeX: {args.out_tex}\n")

if __name__ == "__main__":
    main()
