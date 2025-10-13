#!/usr/bin/env python3
"""
LLM Supplementary Experiment Runner:
- Parses /Users/akira/Kantian/prompt.txt (30 prompts = 10Q × {C0,C1,C2})
- Calls GPT-5 via OpenAI Responses API
- Parses "Answer / Confidence / Reason"
- Scores accuracy and overconfidence (wrong & conf >= threshold)
- Saves results CSV and prints summary

Usage:
  python run_llm_experiment.py \
    --prompt-file /Users/akira/Kantian/prompt.txt \
    --model gpt-5 \
    --temperature 0.2 \
    --top-p 1.0 \
    --conf-threshold 70 \
    --out-csv results_llm_experiment.csv

Requirements:
  pip install openai pandas python-dotenv
  export OPENAI_API_KEY=sk-...
"""

import os, re, csv, argparse, math, json
import openai
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

# --- OpenAI Responses API (official SDK) ---
# Docs: https://platform.openai.com/docs/guides/migrate-to-responses
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a consistent scientific reasoner. "
    "Always output exactly three lines with this format:\n"
    "Answer: <short answer>\nConfidence: <0-100 integer>\nReason: <one short sentence>."
)

# ------------------------------------------------------------
# Parsing prompt.txt  (expects Qn title lines, then C0/C1/C2 lines)
# ------------------------------------------------------------
Q_HEADER_RE = re.compile(r"^Q(\d+)\.\s*(.+)$")
COND_RE = re.compile(r"^(C[012]):\s*(.+)$")

@dataclass
class Trial:
    qid: str         # e.g., "Q1"
    domain: str      # "general", "logic", "reading"
    condition: str   # "C0"|"C1"|"C2"
    prompt: str

def parse_prompt_file(path: str) -> List[Trial]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    trials: List[Trial] = []
    domain = ""
    cur_qid = ""
    # crude domain detection by section headers in file
    def current_domain_from_line(ln: str, cur_domain: str) -> str:
        if ln.strip().startswith("A. General knowledge"):
            return "general"
        if ln.strip().startswith("B. Logical"):
            return "logic"
        if ln.strip().startswith("C. Reading"):
            return "reading"
        return cur_domain

    for i, ln in enumerate(lines):
        domain = current_domain_from_line(ln, domain)
        m_q = Q_HEADER_RE.match(ln.strip())
        if m_q:
            cur_qid = f"Q{m_q.group(1)}"
            continue
        m_c = COND_RE.match(ln.strip())
        if m_c and cur_qid:
            cond = m_c.group(1)
            pr = m_c.group(2).strip()
            trials.append(Trial(qid=cur_qid, domain=domain, condition=cond, prompt=pr))

    # Sanity: should be 30 trials (10 * 3)
    if len(trials) != 30:
        print(f"[WARN] Parsed {len(trials)} trials (expected 30). Check prompt file formatting.")
    return trials

# ------------------------------------------------------------
# Gold answers & scoring logic (tolerances + simple patterns)
# ------------------------------------------------------------
# For each Q, define a checker that returns (is_correct: bool)
def is_numeric_close(val: float, target: float, rel_tol: float = 0.03, abs_tol: float = 10.0) -> bool:
    # relative or absolute tolerance
    return abs(val - target) <= max(abs_tol, rel_tol * abs(target))

def extract_number(text: str) -> Optional[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums: return None
    try:
        return float(nums[0])
    except:
        return None

def check_Q1(answer: str) -> bool:
    # Mozart birth year = 1756
    y = extract_number(answer)
    return y is not None and abs(int(y) - 1756) <= 0

def check_Q2(answer: str) -> bool:
    # Mt Fuji ≈ 3776 m
    n = extract_number(answer)
    return n is not None and is_numeric_close(n, 3776, rel_tol=0.03, abs_tol=50)

def check_Q3(answer: str) -> bool:
    # speed of light ≈ 299792 km/s (accept 299000–300000)
    n = extract_number(answer)
    if n is None: return False
    return 299000 <= n <= 300000

def check_Q4(answer: str) -> bool:
    # Capital of Canada = Ottawa
    return "ottawa" in answer.lower()

def check_Q5(answer: str) -> bool:
    # 80 km/h * 2 h = 160 km
    n = extract_number(answer)
    return n is not None and is_numeric_close(n, 160, rel_tol=0.02, abs_tol=2)

def check_Q6(answer: str) -> bool:
    # half of quarter = 1/8 (accept "0.125", "1/8", "one eighth")
    a = answer.lower().replace(" ", "")
    return ("1/8" in a) or ("oneeighth" in a) or ("0.125" in a) or ("0,125" in a)

def check_Q7(answer: str) -> bool:
    # Monday + 9 days = Wednesday
    return "wednesday" in answer.lower()

def check_Q8(answer: str) -> bool:
    # Tokyo -> Shin-Osaka => most likely (Shinkansen) "train"
    return "train" in answer.lower()

def check_Q9(answer: str) -> bool:
    # cause: it fell (falling off table)
    a = answer.lower()
    return ("fell" in a) or ("fall" in a) or ("falling" in a)

def check_Q10(answer: str) -> bool:
    # Logic: cannot conclude all mammals are cats (answer: No)
    a = answer.lower()
    return ("no" in a) or ("cannot" in a) or ("not" in a and "all mammals are cats" in a)

CHECKERS = {
    "Q1": check_Q1, "Q2": check_Q2, "Q3": check_Q3, "Q4": check_Q4,
    "Q5": check_Q5, "Q6": check_Q6, "Q7": check_Q7, "Q8": check_Q8,
    "Q9": check_Q9, "Q10": check_Q10,
}

# ------------------------------------------------------------
# OpenAI call (Responses API)
# ------------------------------------------------------------
_SAMPLING_WARNING_EMITTED = False


def call_model(client: OpenAI, model: str, temperature: float, top_p: float, user_prompt: str, seed: Optional[int] = None) -> str:
    """
    Returns raw text output. We use Responses API (recommended).
    """
    # Docs: Responses API is recommended over legacy chat completions.
    # https://platform.openai.com/docs/guides/migrate-to-responses
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + "\n\nOutput format:\nAnswer: <text>\nConfidence: <0-100>\nReason: <one sentence>"}
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        resp = client.responses.create(**payload)
    except openai.BadRequestError as err:
        global _SAMPLING_WARNING_EMITTED
        message = getattr(err, "message", str(err))
        unsupported_sampling = (
            "Unsupported parameter" in message
            and ("temperature" in message or "top_p" in message)
        )
        if unsupported_sampling:
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            if not _SAMPLING_WARNING_EMITTED:
                print(
                    "[warn] Model does not support sampling parameters "
                    "(temperature/top_p); retrying with provider defaults.",
                    flush=True,
                )
                _SAMPLING_WARNING_EMITTED = True
            resp = client.responses.create(**payload)
        else:
            raise

    # The simplest extraction: first output text
    if resp.output and len(resp.output) > 0 and hasattr(resp.output[0], "content"):
        # SDK v1 returns output list with message parts; get text concatenated
        parts = resp.output[0].content
        if parts and len(parts) > 0 and parts[0].type == "output_text":
            return parts[0].text
    # Fallback: try .output_text (depending on SDK)
    if hasattr(resp, "output_text"):
        return resp.output_text
    # Last resort, stringify
    return str(resp)

# ------------------------------------------------------------
# Parse "Answer / Confidence / Reason"
# ------------------------------------------------------------
ACCEPT_CONF_RE = re.compile(r"confidence\s*:\s*([0-9]{1,3})", re.I)
ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.I)
REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.I)

def parse_acr_block(text: str) -> Tuple[str, Optional[int], str]:
    ans = ""
    conf = None
    rea = ""
    for line in text.splitlines():
        m1 = ANSWER_RE.match(line.strip())
        m2 = ACCEPT_CONF_RE.match(line.strip())
        m3 = REASON_RE.match(line.strip())
        if m1: ans = m1.group(1).strip()
        if m2:
            try:
                conf = int(m2.group(1))
                conf = max(0, min(100, conf))
            except: pass
        if m3: rea = m3.group(1).strip()
    return ans, conf, rea

# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default="/Users/akira/Kantian/prompt.txt")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--conf-threshold", type=int, default=70,
                    help="Overconfidence if wrong AND confidence >= this")
    ap.add_argument("--out-csv", default="results_llm_experiment.csv")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    trials = parse_prompt_file(args.prompt_file)
    client = OpenAI()

    rows = []
    for tr in trials:
        # Build a single-turn prompt: just the question variant; system carries format instruction
        raw = call_model(
            client=client,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            user_prompt=tr.prompt,
            seed=args.seed
        )
        ans, conf, rea = parse_acr_block(raw)

        # Score
        checker = CHECKERS.get(tr.qid)
        correct = None
        if checker:
            correct = 1 if checker(ans) else 0
        else:
            correct = None

        overconf = None
        if correct is not None and conf is not None:
            overconf = int((correct == 0) and (conf >= args.conf_threshold))

        rows.append({
            "qid": tr.qid,
            "domain": tr.domain,
            "condition": tr.condition,
            "prompt": tr.prompt,
            "raw_output": raw,
            "answer": ans,
            "confidence": conf,
            "reason": rea,
            "correct": correct,
            "overconfident": overconf
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}  ({len(df)} rows)")

    # ---- Summary
    def safe_mean(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if len(s) else float("nan")

    # Per-condition aggregates
    summary = []
    for cond in ["C0","C1","C2"]:
        d = df[df["condition"] == cond]
        n = len(d)
        acc = safe_mean(d["correct"]) if n else float("nan")
        mc_all = safe_mean(d["confidence"])
        mc_correct = safe_mean(d[d["correct"]==1]["confidence"])
        mc_wrong = safe_mean(d[d["correct"]==0]["confidence"])
        over = safe_mean(d["overconfident"])
        summary.append({
            "condition": cond,
            "n": n,
            "accuracy": round(acc, 3) if not math.isnan(acc) else None,
            "mean_conf_all": round(mc_all, 1) if not math.isnan(mc_all) else None,
            "mean_conf_correct": round(mc_correct, 1) if not math.isnan(mc_correct) else None,
            "mean_conf_wrong": round(mc_wrong, 1) if not math.isnan(mc_wrong) else None,
            "overconfidence_rate": round(over, 3) if not math.isnan(over) else None
        })
    print("\n=== Summary by condition ===")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))

    # Simple overall summary
    acc_overall = safe_mean(df["correct"])
    over_all = safe_mean(df["overconfident"])
    print("\n=== Overall ===")
    print(json.dumps({
        "n_total": len(df),
        "accuracy_overall": round(acc_overall,3) if not math.isnan(acc_overall) else None,
        "overconfidence_rate_overall": round(over_all,3) if not math.isnan(over_all) else None
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
