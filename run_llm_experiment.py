#!/usr/bin/env python3
"""
LLM Supplementary Experiment Runner (Kantian-ready)

Overview:
- Parses a prompt file in the Kantian ABC format (default: ./prompt.txt)
  - A/B/C layers with items like: A001., B001., C001.
  - Each item has variants C0, C1, C2 (three conditions per item)
- Calls the model via OpenAI Responses API
- Enforces a strict output format and extracts: Answer / Confidence / Reason
- Computes per-condition summaries (accuracy if checkers exist; otherwise confidence-only)
- Saves per-trial results to CSV and prints a concise summary
- Automatically invokes Kantian post-evaluation at the end

Default behavior (no arguments):
  python run_llm_experiment.py
  → reads ./prompt.txt
  → writes ./results_llm_experiment.csv
  → prints progress like: [1/300] A001 C0 :: general
  → runs kantian_eval.run_default() which writes ./results_kantian_eval.csv

Optional arguments (still supported):
  --prompt-file <path>      # default: ./prompt.txt
  --model <name>            # default: gpt-5
  --temperature <float>     # default: 0.2
  --top-p <float>           # default: 1.0
  --conf-threshold <int>    # default: 70 (flag overconfidence when wrong & >= threshold)
  --out-csv <path>          # default: results_llm_experiment.csv
  --seed <int>              # default: 2025 (passed to API when supported)

Notes:
- Legacy Q-format (Q1. … + C0/C1/C2) is still supported.
- Accuracy metrics require task-specific checkers; ABC items are summarized mainly by confidence unless checkers are added.
- Requires: openai, pandas, python-dotenv; env var OPENAI_API_KEY must be set.

Non-arbitrariness commitments:
- Label inheritance → C0/C1/C2 share the same ground-truth for A-layer.
- Fixed normalization → normalize_yes_no() is public and unchanged across runs.
- Report denominators → exclusion rates are printed per condition and overall.
- Avoid pseudo-independence → interpret condition differences item-wise; use paired tests if added.
- Version pinning → commit/DOI for prompt.txt; seed/config logged.
- No leakage → model input excludes any Label lines.
"""

import os, re, csv, argparse, math, json
import sys
import openai
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import traceback

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
ABC_ID_RE   = re.compile(r"^([ABC])(\d{3})\.\s*$")
COND_RE     = re.compile(r"^(C[012]):\s*(.+)$")

@dataclass
class Trial:
    qid: str         # e.g., "Q1", "A001"
    domain: str      # "general", "logic", "reading" (mapped) or semantic|reflective|meta upstream
    condition: str   # "C0"|"C1"|"C2"
    prompt: str
    label: Optional[int] = None  # ground-truth for A-layer (inherits from C0)

def parse_prompt_file(path: str) -> List[Trial]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    trials: List[Trial] = []
    domain = ""
    cur_qid = ""
    # domain detection for both legacy and Kantian formats
    def current_domain_from_line(ln: str, cur_domain: str) -> str:
        s = ln.strip()
        # Legacy demo
        if s.startswith("A. General knowledge"):
            return "general"
        if s.startswith("B. Logical"):
            return "logic"
        if s.startswith("C. Reading"):
            return "reading"
        # Kantian v2 generator headers
        if s.startswith("A. Semantic/Factual"):
            return "semantic"
        if s.startswith("B. Reflective/Consistency"):
            return "reflective"
        if s.startswith("C. Meta-epistemic/Limits"):
            return "meta"
        return cur_domain

    # Detect format: Q-format or Kantian ABC-format
    text_all = "\n".join(lines)
    is_kantian_abc = bool(re.search(r"^A\.\s*Semantic/Factual", text_all, flags=re.M))

    label_map: Dict[str, Optional[int]] = {}

    if is_kantian_abc:
        # Parse Kantian ABC format: IDs like A001. then C0/C1/C2 lines, optional Label below
        for i, ln in enumerate(lines):
            domain = current_domain_from_line(ln, domain)
            m_id = ABC_ID_RE.match(ln.strip())
            if m_id:
                cur_qid = f"{m_id.group(1)}{m_id.group(2)}"
                continue
            m_c = COND_RE.match(ln.strip())
            if m_c and cur_qid:
                cond = m_c.group(1)
                pr = m_c.group(2).strip()
                # Map domains to legacy names for downstream summaries
                dom_map = {"semantic":"general", "reflective":"logic", "meta":"reading"}
                trials.append(Trial(qid=cur_qid, domain=dom_map.get(domain, domain or "general"), condition=cond, prompt=pr))
                # continue to next line
            # Capture labels: lines like "Label (for C0): 0" appear right after A-layer blocks
            if ln.strip().lower().startswith("label"):
                try:
                    val = int(ln.split(":")[1].strip())
                except Exception:
                    val = None
                if cur_qid:
                    label_map[cur_qid] = val
                continue
        # Assign labels to A-layer trials (inherit across C0/C1/C2)
        if label_map:
            for i, t in enumerate(trials):
                if t.qid and t.qid.startswith("A") and t.qid in label_map:
                    trials[i].label = label_map[t.qid]
        # Sanity check (expected 300 = 100 * 3, but allow any multiple of 3)
        if len(trials) % 3 != 0 or len(trials) == 0:
            print(f"[WARN] Parsed {len(trials)} trials from Kantian ABC format (expected multiple of 3). Check formatting.")
        return trials
    else:
        # Legacy Q-format: Qn. header then C0/C1/C2 variants
        def current_domain_from_line_legacy(ln: str, cur_domain: str) -> str:
            if ln.strip().startswith("A. General knowledge"):
                return "general"
            if ln.strip().startswith("B. Logical"):
                return "logic"
            if ln.strip().startswith("C. Reading"):
                return "reading"
            return cur_domain

        domain = ""
        cur_qid = ""
        for i, ln in enumerate(lines):
            domain = current_domain_from_line_legacy(ln, domain)
            m_q = Q_HEADER_RE.match(ln.strip())
            if m_q:
                cur_qid = f"Q{m_q.group(1)}"
                continue
            m_c = COND_RE.match(ln.strip())
            if m_c and cur_qid:
                cond = m_c.group(1)
                pr = m_c.group(2).strip()
                trials.append(Trial(qid=cur_qid, domain=domain, condition=cond, prompt=pr))

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
_SEED_SUPPORTED = True
_SEED_WARN_EMITTED = False
_SEED_NOTE_EMITTED = False


def call_model(client: OpenAI, model: str, temperature: float, top_p: float, user_prompt: str, seed: Optional[int] = None) -> str:
    """
    Returns raw text output. We use Responses API (recommended).
    """
    global _SAMPLING_WARNING_EMITTED, _SEED_SUPPORTED, _SEED_WARN_EMITTED, _SEED_NOTE_EMITTED
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

    want_seed = seed
    tried_fallback = False

    def _create(include_seed: bool):
        p = dict(payload)
        if include_seed and (want_seed is not None):
            p["seed"] = want_seed
        return client.responses.create(**p)

    try:
        if _SEED_SUPPORTED and (want_seed is not None):
            resp = _create(include_seed=True)
        else:
            resp = _create(include_seed=False)
    except TypeError as err:
        if "unexpected keyword argument" in str(err) and "seed" in str(err):
            _SEED_SUPPORTED = False
            if not _SEED_WARN_EMITTED:
                print("[warn] SDK does not accept 'seed'; retrying without it.", flush=True)
                _SEED_WARN_EMITTED = True
            try:
                resp = _create(include_seed=False)
            except Exception as err2:
                msg2 = getattr(err2, "message", str(err2))
                if ("Unsupported parameter" in msg2) and ("temperature" in msg2 or "top_p" in msg2):
                    payload.pop("temperature", None)
                    payload.pop("top_p", None)
                    if not _SAMPLING_WARNING_EMITTED:
                        print("[warn] Model does not support sampling parameters; retrying without temperature/top_p.", flush=True)
                        _SAMPLING_WARNING_EMITTED = True
                    resp = _create(include_seed=False)
                else:
                    if any(k in msg2.lower() for k in ["model not found", "unknown model", "does not exist", "is not available", "unsupported model"]):
                        payload["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                        print(f"[warn] Falling back to model: {payload['model']}", flush=True)
                        tried_fallback = True
                        resp = _create(include_seed=False)
                    else:
                        raise RuntimeError(f"Responses API call failed without seed: {msg2}")
        else:
            raise
    except Exception as err:
        msg = getattr(err, "message", str(err))
        if ("Unsupported parameter" in msg) and ("temperature" in msg or "top_p" in msg):
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            if not _SAMPLING_WARNING_EMITTED:
                print("[warn] Model does not support sampling parameters; retrying without temperature/top_p.", flush=True)
                _SAMPLING_WARNING_EMITTED = True
            resp = _create(include_seed=False)
        else:
            if any(k in msg.lower() for k in ["model not found", "unknown model", "does not exist", "is not available", "unsupported model"]):
                payload["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                print(f"[warn] Falling back to model: {payload['model']}", flush=True)
                tried_fallback = True
                resp = _create(include_seed=False)
            else:
                print("[error] Responses API call failed:\n" + msg, flush=True)
                raise
    # one-time note after first seed disable
    if not _SEED_SUPPORTED and not _SEED_NOTE_EMITTED and (want_seed is not None):
        print("[note] Seed not supported in this SDK; reproducibility relies on fixed prompts/temperature.", flush=True)
        _SEED_NOTE_EMITTED = True

    # The simplest extraction: first output text
    if resp.output and len(resp.output) > 0 and hasattr(resp.output[0], "content"):
        # SDK v1 returns output list with message parts; get text concatenated
        parts = resp.output[0].content
        if parts and len(parts) > 0 and parts[0].type == "output_text":
            if tried_fallback:
                print("[info] Used fallback model for this call.", flush=True)
            return parts[0].text
    # Fallback: try .output_text (depending on SDK)
    if hasattr(resp, "output_text"):
        if tried_fallback:
            print("[info] Used fallback model for this call.", flush=True)
        return resp.output_text
    # Last resort, stringify
    if tried_fallback:
        print("[info] Used fallback model for this call.", flush=True)
    return str(resp)

# ------------------------------------------------------------
# Parse "Answer / Confidence / Reason"
# ------------------------------------------------------------
ACCEPT_CONF_RE = re.compile(r"confidence\s*:\s*([0-9]{1,3})", re.I)
ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.I)
REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.I)

# ------------------------------------------------------------
# Parse "Answer / Confidence / Reason"
# ------------------------------------------------------------
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
# Normalize yes/no answers for A-layer label-based scoring
# ------------------------------------------------------------
YES_TOKENS = {"yes","y","true","correct","affirmative","sure","indeed"}
NO_TOKENS  = {"no","n","false","incorrect","negative","nope"}

def normalize_yes_no(text: str) -> Optional[int]:
    """
    Fixed, public rule (non-arbitrary):
    - Map common yes/no tokens (case-insensitive).
    - If mixed or absent, return None (excluded from accuracy).
    """
    if not text:
        return None
    t = text.strip().lower()
    # pick first tokenized word to avoid matching inside reason sentences
    first = re.split(r"[^a-z]+", t, maxsplit=1)[0]
    if first in YES_TOKENS:
        return 1
    if first in NO_TOKENS:
        return 0
    # also allow leading phrases like "it is true"/"it is false"
    if "it is true" in t or "that is true" in t:
        return 1
    if "it is false" in t or "that is false" in t:
        return 0
    return None

# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default=os.path.join(os.getcwd(), "prompt.txt"))
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--conf-threshold", type=int, default=70,
                    help="Overconfidence if wrong AND confidence >= this")
    ap.add_argument("--out-csv", default="results_llm_experiment.csv")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--force-run", action="store_true", help="Ignore existing results and call the API")
    ap.add_argument("--reuse-only", action="store_true", help="Do not call the API; summarize existing results CSV and exit")
    ap.add_argument("--no-post-eval", action="store_true", help="Skip calling kantian_eval.run_default() at the end")
    args = ap.parse_args()
    print(f"[cfg] prompt_file={args.prompt_file}  model={args.model}  temperature={args.temperature}  top_p={args.top_p}  seed={args.seed}")
    if args.model == "gpt-5":
        print("[note] If this model is unavailable in your account, the runner will fall back to OPENAI_MODEL or gpt-4o-mini.")

    out_csv = getattr(args, "out_csv", "results_llm_experiment.csv")

    # --- REUSE EXISTING RESULTS FAST-PATH ---
    if args.reuse_only or (os.path.exists(out_csv) and not args.force_run):
        if os.path.exists(out_csv):
            print(f"[reuse] Using existing results: {out_csv}", flush=True)
            try:
                df = pd.read_csv(out_csv)
            except Exception as e:
                print(f"[reuse][error] Failed to read existing CSV: {e}", flush=True)
                if args.reuse_only:
                    sys.exit(1)
            else:
                # Simple condition-wise summary if columns exist
                print("\n=== Summary by condition (reused) ===", flush=True)
                if "condition" in df.columns:
                    for cond, g in df.groupby("condition"):
                        entry = {"condition": str(cond), "n": int(len(g))}
                        if "correct" in g.columns:
                            try:
                                entry["accuracy"] = round(float(pd.to_numeric(g["correct"], errors="coerce").mean()), 3)
                            except Exception:
                                entry["accuracy"] = None
                        if "confidence" in g.columns:
                            try:
                                entry["mean_conf_all"] = round(float(pd.to_numeric(g["confidence"], errors="coerce").mean()), 1)
                            except Exception:
                                entry["mean_conf_all"] = None
                        if "overconfident" in g.columns:
                            try:
                                entry["overconfidence_rate"] = round(float(pd.to_numeric(g["overconfident"], errors="coerce").mean()), 3)
                            except Exception:
                                entry["overconfidence_rate"] = None
                        print(json.dumps(entry, ensure_ascii=False), flush=True)
                else:
                    print("[reuse] condition column not found; printed row count only.", flush=True)
                print("\n[reuse] Skipping API calls; done.", flush=True)
                sys.exit(0)
        else:
            print(f"[reuse][error] No existing CSV found at {out_csv}. Nothing to summarize.", flush=True)
            sys.exit(1)
    # --- END REUSE FAST-PATH ---

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    trials = parse_prompt_file(args.prompt_file)
    client = OpenAI()

    rows = []
    excl_counts = {"C0":0, "C1":0, "C2":0, "ALL":0}  # exclusions due to ambiguous answers under label-based scoring
    total = len(trials)
    for idx, tr in enumerate(trials, start=1):
        print(f"[{idx}/{total}] {tr.qid} {tr.condition} :: {tr.domain}", flush=True)
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
            # Label-based scoring for A-layer (label inherited from C0)
            if tr.label in (0,1):
                yn = normalize_yes_no(ans)
                if yn is None:
                    correct = None  # excluded
                    excl_counts[tr.condition] += 1
                    excl_counts["ALL"] += 1
                else:
                    correct = 1 if yn == tr.label else 0
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
        excl = excl_counts.get(cond, 0)
        excl_rate = round(excl / n, 3) if n else None
        summary.append({
            "condition": cond,
            "n": n,
            "accuracy": round(acc, 3) if not math.isnan(acc) else None,
            "mean_conf_all": round(mc_all, 1) if not math.isnan(mc_all) else None,
            "mean_conf_correct": round(mc_correct, 1) if not math.isnan(mc_correct) else None,
            "mean_conf_wrong": round(mc_wrong, 1) if not math.isnan(mc_wrong) else None,
            "overconfidence_rate": round(over, 3) if not math.isnan(over) else None,
            "exclusion_rate": excl_rate
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
        "overconfidence_rate_overall": round(over_all,3) if not math.isnan(over_all) else None,
        "exclusion_rate_overall": round(excl_counts["ALL"]/len(df),3) if len(df) else None
    }, ensure_ascii=False))

    print("\n[info] Non-arbitrariness commitments:")
    print("- Label inheritance: C0/C1/C2 are framing variants of the same proposition; A-layer labels are inherited across conditions.")
    print("- Ambiguous answers: normalize_yes_no() is fixed and public; non-matching answers are excluded from accuracy.")
    print("- Denominator differences: we report exclusion rates per condition and overall.")
    print("- Pseudo-independence: compare conditions item-wise; paired analyses can be added if needed.")
    print("- Versioning: pin prompt.txt in Git/Zenodo; seed and config are logged at start.")
    print("- No data leakage: the model never receives Label lines; only the C* prompt text is sent.")

    # --- Kantian Evaluation Auto-call (run only when API path executed) ---
    if not args.no_post_eval and os.getenv("KANTIAN_SKIP_EVAL") not in ("1","true","TRUE","yes","YES"):
        try:
            print("[Kantian] starting evaluation ...", flush=True)
            import kantian_eval
            kantian_eval.run_default()
            print("[Kantian] evaluation complete", flush=True)
        except Exception as e:
            print(f"[Kantian] evaluation failed: {e}", flush=True)
    else:
        print("[Kantian] post-evaluation skipped.", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
