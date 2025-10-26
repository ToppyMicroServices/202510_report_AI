# kantian_eval.py
import os, json, time, csv, argparse, sys, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

# ========= Metrics =========
def brier(p: float, y: int) -> float:
    p = min(max(float(p), 0.0), 1.0)
    return (p - y) ** 2

def ece(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    probs = np.clip(np.array(probs, dtype=float), 0, 1)
    labels = np.array(labels, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i+1] if i < n_bins-1 else probs <= bins[i+1])
        if idx.sum() == 0:
            continue
        conf = probs[idx].mean()
        acc = labels[idx].mean()
        ece_val += (idx.sum() / len(probs)) * abs(acc - conf)
    return float(ece_val)

# ========= Prompt Templates (固定様式) =========
SYSTEM_HEADER = (
    "You are a careful assistant. Follow the output schema strictly. "
    "If uncertain, state uncertainty explicitly."
)

PROMPT_TEMPLATE = """\
[ROLE]
You answer one question, then (separately) provide a short self-critique, then provide a revised answer.

[IO FORMAT]
Return a single JSON object with keys:
- "answer0": string
- "critique": {{
    "factual_consistency": string,
    "logical_coherence": string,
    "uncertainty_statement": string
  }},
- "answer1": string
- "probability_yes": float
- "confidence_note": string

[QUESTION]
{question}

[DECISION TASK]
Binary target: y ∈ {{0,1}}. "probability_yes" must be your belief that y=1.

[CONSTRAINTS]
- Use plain text (no markdown) inside JSON values.
- Do not add keys. Do not add commentary outside JSON.
"""

# ========= LLM Caller (OpenAI互換) =========
def call_llm_openai(prompt: str, model: str, temperature: float, seed: Optional[int]) -> str:
    # OpenAI SDK v1系を想定（環境に合わせて微調整）
    from openai import OpenAI
    client = OpenAI()
    kwargs = dict(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_HEADER},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    if seed is not None:
        kwargs["seed"] = seed  # 未対応モデルは無視される
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

# ========= Data Structures =========
@dataclass
class Trial:
    prompt_id: str
    run_idx: int
    seed: Optional[int]
    question: str
    y_true: Optional[int]
    answer0: str
    critique_factual: str
    critique_logic: str
    critique_uncert: str
    answer1: str
    p_yes: float
    brier1: Optional[float]
    t_ms: int

# ========= Core Evaluator =========
class KantianEvaluator:
    def __init__(self, model: str, temperature: float, repeats: int, use_seed: bool):
        self.model = model
        self.temperature = temperature
        self.repeats = repeats
        self.use_seed = use_seed
        self.trials: List[Trial] = []

    def _one_call(self, question: str, seed: Optional[int]) -> Dict[str, Any]:
        raw = call_llm_openai(
            prompt=PROMPT_TEMPLATE.format(question=question),
            model=self.model,
            temperature=self.temperature,
            seed=seed
        )
        try:
            obj = json.loads(raw)
        except Exception:
            # JSON崩れは規約的にp=0.5扱い（事前に固定）
            obj = {
                "answer0": "",
                "critique": {
                    "factual_consistency": "parse_error",
                    "logical_coherence": "parse_error",
                    "uncertainty_statement": "parse_error",
                },
                "answer1": "",
                "probability_yes": 0.5,
                "confidence_note": "fallback_due_to_parse_error"
            }
        return obj

    def evaluate_items(self, items: List[Dict[str, Any]]):
        for it in items:
            pid = it["id"]
            q = it["question"]
            y = it.get("label", None)
            for r in range(self.repeats):
                seed = r if self.use_seed else None
                t0 = time.time()
                obj = self._one_call(q, seed)
                t1 = time.time()

                ans0 = (obj.get("answer0") or "").strip()
                crt  = obj.get("critique") or {}
                ans1 = (obj.get("answer1") or "").strip()
                p    = float(obj.get("probability_yes", 0.5))

                b1 = None
                if y is not None:
                    b1 = brier(p, int(y))

                tr = Trial(
                    prompt_id=pid, run_idx=r, seed=seed, question=q, y_true=y,
                    answer0=ans0,
                    critique_factual=crt.get("factual_consistency",""),
                    critique_logic=crt.get("logical_coherence",""),
                    critique_uncert=crt.get("uncertainty_statement",""),
                    answer1=ans1, p_yes=p, brier1=b1,
                    t_ms=int((t1 - t0)*1000)
                )
                self.trials.append(tr)

    def write_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "prompt_id","run_idx","seed","question","y_true",
                "answer0","critique_factual","critique_logic","critique_uncert",
                "answer1","p_yes","brier1","latency_ms"
            ])
            for t in self.trials:
                w.writerow([
                    t.prompt_id, t.run_idx, t.seed, t.question, t.y_true,
                    t.answer0, t.critique_factual, t.critique_logic, t.critique_uncert,
                    t.answer1, f"{t.p_yes:.6f}", 
                    (f"{t.brier1:.6f}" if t.brier1 is not None else ""),
                    t.t_ms
                ])

    def summarize(self) -> Dict[str, Any]:
        # ECE/Brier集約（ラベルあるもののみ）
        ps, ys, bs = [], [], []
        for t in self.trials:
            if t.y_true is not None:
                ys.append(int(t.y_true))
                ps.append(float(t.p_yes))
                if t.brier1 is not None:
                    bs.append(float(t.brier1))
        out: Dict[str, Any] = {
            "n_trials": len(self.trials),
            "n_labeled": len(ys),
            "temperature": self.temperature,
            "repeats": self.repeats,
            "model": self.model,
        }
        if len(ys) > 0:
            out["ece_10bins"] = ece(ps, ys, n_bins=10)
            out["brier_mean"] = float(np.mean(bs)) if bs else None
        return out

# ========= CLI =========
def load_items_from_path(path: str) -> List[Dict[str, Any]]:
    """
    入力形式（いずれか）:
      - JSONL: {"id":"q001","question":"...","label":1}
      - CSV:   id,question,label
    """
    items: List[Dict[str, Any]] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append({"id":obj["id"], "question":obj["question"], "label":obj.get("label", None)})
    elif path.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            items.append({"id": str(row["id"]), "question": str(row["question"]), "label": row.get("label", None)})
    elif path.lower().endswith(".txt"):
        import re
        items = []
        current_id, current_label, current_c0 = None, None, None
        last_index_by_id = {}  # track index of last appended item per id for later label update

        with open(path, encoding="utf-8") as f:
            for line in f:
                # Match IDs like A001. / B001. / C001.
                if m := re.match(r"^(A|B|C)(\d{3})\.\s*$", line):
                    current_id = m.group(0).rstrip(".")
                    current_c0 = None
                    current_label = None
                    continue

                # Capture the C0 variant line as the evaluable question
                if line.startswith("C0: "):
                    current_c0 = line.replace("C0: ", "").strip()
                    if current_id and current_c0:
                        # Append immediately with label=None (for B/C). A-layer will be updated when Label appears.
                        items.append({"id": current_id, "question": current_c0, "label": None})
                        last_index_by_id[current_id] = len(items) - 1
                    continue

                # If a Label line appears (A-layer), update the most recent item for this id
                if line.startswith("Label"):
                    try:
                        current_label = int(line.split(":")[1].strip())
                    except Exception:
                        current_label = None
                    if current_id and current_id in last_index_by_id:
                        idx = last_index_by_id[current_id]
                        items[idx]["label"] = current_label
                    continue

        return items
    else:
        raise ValueError("Unsupported prompt file. Use .jsonl, .csv, or .txt (Kantian format)")
    return items

def main():
    ap = argparse.ArgumentParser(description="Kantian Stability Evaluator (Prompt--Critique--Revision, non-arbitrary)")
    ap.add_argument("--prompts", default=os.path.join(os.getcwd(), "prompt.txt"), help="Path to prompts file (default: ./prompt.txt)")
    ap.add_argument("--out_csv", default="results_kantian_eval.csv", help="Output trial CSV")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-seed", action="store_true", help="Do not set seed per repeat")
    args = ap.parse_args()

    # 事前固定（恣意性防止）：top_p, penalties等は本ファイルでは操作しない方針
    random.seed(0); np.random.seed(0)

    items = load_items_from_path(args.prompts)
    ke = KantianEvaluator(
        model=args.model,
        temperature=args.temperature,
        repeats=args.repeats,
        use_seed=(not args.no_seed)
    )
    ke.evaluate_items(items)
    ke.write_csv(args.out_csv)
    summary = ke.summarize()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def run_default():
    """
    Programmatic entrypoint used by run_llm_experiment.py.
    Runs the evaluator with default CLI settings (prompts=./prompt.txt,
    outputs results_kantian_eval.csv).
    """
    prev_argv = list(sys.argv)
    try:
        # Minimal argv so argparse picks defaults
        sys.argv = ["kantian_eval.py"]
        main()
    finally:
        sys.argv = prev_argv

if __name__ == "__main__":
    main()
    