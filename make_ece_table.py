#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ece_table.py
- モデル×プロンプト別に各種指標を集計（mean, count, 95% CI）し、
  CSV と LaTeX (booktabs) を出力するスクリプト。

依存: pandas
デフォルト:
  入力:  results_llm_experiment.csv
  出力:  summary_by_model_prompt.csv, Tables/tab_ece_model_prompt.tex
  ％表示: accuracy, hallucination_rate

usage 例:
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
    ap = argparse.ArgumentParser(description="Aggregate LLM experiment metrics by model × prompt.")
    ap.add_argument(
        "csv",
        nargs="?",
        default="results_llm_experiment.csv",
        help="Input CSV (default: results_llm_experiment.csv)",
    )
    ap.add_argument("--out-csv", default="summary_by_model_prompt.csv", help="Output aggregated CSV path")
    ap.add_argument("--out-tex", default="Tables/tab_ece_model_prompt.tex", help="Output LaTeX table path")
    # 明示指定（未指定なら自動検出）
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--prompt-col", default=None)
    ap.add_argument("--ece-eqwidth", default=None)
    ap.add_argument("--ece-eqfreq", default=None)
    ap.add_argument("--brier", default=None)
    ap.add_argument("--logloss", default=None)
    ap.add_argument("--accuracy", default=None)
    ap.add_argument("--halluc", default=None, help="hallucination rate column")
    # 表示系
    ap.add_argument("--digits", type=int, default=3, help="LaTeX数値の小数桁")
    ap.add_argument(
        "--as-perc",
        nargs="*",
        default=["accuracy", "hallucination_rate"],
        help="％表示にする列名（accuracy 等）",
    )
    ap.add_argument("--title", default=None, help="LaTeX表題（未指定ならキャプション固定文）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # 列名マップ（小文字→オリジナル）
    lower_map = {c.lower().strip(): c for c in df.columns}

    # model / prompt 列（明示優先→自動検出）
    model_col = args.model_col or detect_column(lower_map, "model", "model_name")
    if not model_col:
        # 最初の列を model と仮定
        model_col = df.columns[0]
    prompt_col = args.prompt_col or detect_column(lower_map, "prompt", "prompt_type", "prompt_name")

    # メトリクス列（明示優先→自動検出（同義語））
    ece_w  = args.ece_eqwidth or detect_column(lower_map, "ece_equal_width", "ece_eqwidth")
    ece_f  = args.ece_eqfreq or detect_column(lower_map, "ece_equal_freq", "ece_eqfreq")
    brier  = args.brier       or detect_column(lower_map, "brier", "brier_score")
    logl   = args.logloss     or detect_column(lower_map, "logloss", "log_loss")
    acc    = args.accuracy    or detect_column(lower_map, "accuracy", "acc")
    hall   = args.halluc      or detect_column(lower_map, "hallucination_rate", "halluc_rate", "hallu_rate")

    # 使う列を並べる（存在するものだけ）
    metric_cols = [(name, col) for name, col in [
        ("ece_equal_width", ece_w),
        ("ece_equal_freq",  ece_f),
        ("brier",           brier),
        ("logloss",         logl),
        ("accuracy",        acc),
        ("hallucination_rate", hall),
    ] if col is not None]

    # 数値列のみ残す（非数は落とす）
    numeric_metrics = []
    for name, col in metric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                continue
        numeric_metrics.append((name, col))

    if not numeric_metrics:
        # 自動フォールバック：model/prompt以外の数値列を最大6個まで採用
        exclude = {model_col}
        if prompt_col: exclude.add(prompt_col)
        fallback = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        if not fallback:
            print("No numeric metric columns could be identified.", file=sys.stderr)
            sys.exit(1)
        numeric_metrics = [(c, c) for c in fallback[:6]]

    # 集計キー
    group_keys = [model_col] + ([prompt_col] if prompt_col else [])
    g = df.groupby(group_keys, dropna=False)

    # mean, std, n と 95%CI（正規近似）を計算
    rows = []
    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {model_col: keys[0]}
        if prompt_col:
            rec[prompt_col] = keys[1]
        n = len(sub)
        rec["N"] = n
        for name, col in numeric_metrics:
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

    # ％表示に変換する列（*_mean / *_ci_low / *_ci_high）
    as_perc = set(s.lower() for s in args.as_perc)
    for name, _col in numeric_metrics:
        if name.lower() in as_perc:
            for suf in ("mean", "ci_low", "ci_high"):
                c = f"{name}_{suf}"
                if c in agg.columns:
                    agg[c] = agg[c] * 100.0

    # CSV 出力
    out_csv_path = Path(args.out_csv)
    if out_csv_path.parent and not out_csv_path.parent.exists():
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv_path, index=False)

    # LaTeX 表の構成（mean のみ表示、N 付き）
    show_names = [name for name, _ in numeric_metrics]  # 表示順
    header = ([ "Model", "Prompt"] if prompt_col else ["Model"]) \
             + [n.replace("_"," ").title() for n in show_names] + ["N"]

    # 列指定（LaTeXの整列）
    align = ("ll" + "r"*len(show_names) + "r") if prompt_col else ("l" + "r"*len(show_names) + "r")

    # LaTeX 行を生成
    lines = []
    lines.append(r"\begin{tabular}{%s}" % align)
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # 並び替え（Model→Prompt）
    sort_cols = [model_col] + ([prompt_col] if prompt_col else [])
    agg_sort = agg.sort_values(sort_cols)

    def pick_mean(row, name):
        val = row.get(f"{name}_mean", np.nan)
        return format_float(val, args.digits)

    for _, row in agg_sort.iterrows():
        vals = [str(row[model_col])]
        if prompt_col:
            vals.append(str(row[prompt_col]))
        for name in show_names:
            vals.append(pick_mean(row, name))
        vals.append(str(int(row["N"])))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # LaTeX 出力
    out_tex = "\n".join(lines)
    # 親ディレクトリが無い場合は作成案内のみ（自動作成はしない）
    out_tex_path = Path(args.out_tex)
    if out_tex_path.parent and not out_tex_path.parent.exists():
        out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text(out_tex, encoding="utf-8")

    # 進捗表示
    sys.stderr.write(f"[ok] wrote CSV: {args.out_csv}\n")
    sys.stderr.write(f"[ok] wrote LaTeX: {args.out_tex}\n")

if __name__ == "__main__":
    main()
