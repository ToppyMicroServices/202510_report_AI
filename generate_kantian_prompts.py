# generate_kantian_prompts.py
import os, random, math, datetime, shutil

EXTERNAL_TEMPLATE_DIR = "/Users/akira/Kantian/templates"
SR_EXTRA_PATH = os.path.join(EXTERNAL_TEMPLATE_DIR, "self_refine_extra.txt")
RX_EXTRA_PATH = os.path.join(EXTERNAL_TEMPLATE_DIR, "reflexion_extra.txt")
CN_EXTRA_PATH = os.path.join(EXTERNAL_TEMPLATE_DIR, "constitutional_extra.txt")

def _load_extra_lines(path: str):
    lines = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    lines.append(s)
    except Exception:
        pass
    return lines

OUT_PATH = "/Users/akira/Kantian/prompt.txt"
random.seed(42)

# Literature-grounded templates (concise paraphrases)
SELF_REFINE = [
    "Critique your previous answer for factual accuracy, logical soundness, and completeness.",
    "Identify specific mistakes, if any, and propose concrete improvements.",
    "Check whether each claimed fact is supported by reliable evidence.",
    "Point out overconfident statements and replace them with uncertainty-aware phrasing.",
    "List missing steps in the reasoning chain and fill them succinctly.",
    "Detect ambiguous terms and redefine them precisely before revising.",
    "Compare the answer against an alternative approach and explain divergences.",
    "Separate observation from inference; flag where you are speculating.",
    "Normalize units, dates, and notations; correct any inconsistencies.",
    "Propose a minimal revision that fixes the main error without adding new ones.",
    "State residual uncertainties after revision and why they remain.",
    "If information is insufficient, recommend what additional data would resolve it.",
    "Cite the most authoritative source you relied on, if any.",
    "Flag any leaps of logic and justify or remove them.",
    "Replace vague qualifiers with quantitative estimates or ranges.",
    "Note any dataset limitations or sampling biases that affect the claim.",
    "Check consistency across all sentences; remove contradictions.",
    "Simplify the explanation while preserving accuracy.",
    "State which parts are normative judgments vs. empirical facts.",
    "Ensure numerical calculations are re-done and verified.",
    "Cross-verify names, dates, and attributions.",
    "Indicate if the conclusion depends on unstated context.",
    "Align terminology with standard definitions in the field.",
    "If multiple answers are plausible, present the top two and why.",
    "Clarify scope and boundary conditions explicitly.",
    "Check for cherry-picking; add a counterexample if relevant.",
    "Make causality claims conditional unless strongly supported.",
    "Remove weasel words and tighten the language.",
    "Turn passive voice into active voice where clarity improves.",
    "Confirm the final answer matches the question exactly.",
]

REFLEXION = [
    "Reflect on your reasoning process: which step could fail under uncertainty?",
    "Re-evaluate assumptions that may not hold universally and state your confidence.",
    "Identify the weakest premise and stress-test it with counterexamples.",
    "Consider edge cases or boundary conditions that might break the argument.",
    "Distinguish data-driven claims from prior beliefs and note possible biases.",
    "Explain how the conclusion would change if a key assumption were false.",
    "Clarify the causal vs. correlational nature of the claim.",
    "Outline an alternative reasoning path and compare expected outcomes.",
    "Quantify uncertainty where possible; otherwise give qualitative ranges.",
    "Flag dependence on time, place, or definitions that may vary across sources.",
    "Note any distribution shift or domain mismatch affecting validity.",
    "Describe a simple test that could falsify your conclusion.",
    "Isolate which sub-claim contributes most to overall uncertainty.",
    "State what evidence would most efficiently reduce uncertainty.",
    "Check whether you inadvertently answered a different question.",
    "Surface any hidden value judgments or utility trade-offs.",
    "Map the argument as premises → inference → conclusion; verify each link.",
    "Contrast top-down (theory-first) vs. bottom-up (data-first) reasoning here.",
    "Indicate how sensitive the conclusion is to measurement error.",
    "Highlight any reliance on authority rather than evidence.",
    "Explain how time horizons alter the recommendation or conclusion.",
    "Note if selection bias or survivorship bias could be present.",
    "Check for circular reasoning or tautologies and remove them.",
    "Consider stakeholder perspectives that might shift the framing.",
]

CONSTITUTIONAL = [
    "If the request conflicts with safety or ethics, politely refuse and briefly explain why.",
    "When information is uncertain, communicate uncertainty clearly rather than guessing.",
    "Prefer verifiable evidence and cite what would change your view.",
    "Avoid giving medical, legal, or financial advice beyond general information; recommend consulting a professional.",
    "Protect personal data and avoid deanonymizing individuals.",
    "Do not fabricate sources; admit when you do not know.",
    "Avoid harmful instructions or facilitation of wrongdoing.",
    "Use the least risky option that still answers the question responsibly.",
    "Disclose model limitations and potential errors when relevant.",
    "Calibrate confidence: avoid overclaiming; choose conservative language when unsure.",
    "Prefer primary sources or high-quality reviews over unsourced claims.",
    "Outline safe fallback actions the user can take when uncertainty is high.",
    "Avoid stereotypes and respect fairness across groups.",
    "Separate informational content from normative recommendations.",
    "Be transparent about data freshness and possible obsolescence.",
    "Encourage user verification for high-stakes decisions.",
    "Preserve context; warn when extrapolating beyond the data.",
    "Avoid dual-use technical detail that enables harm.",
    "Prefer consent-based data handling and minimize data exposure.",
    "State when external expert review would be prudent.",
]

# --- Optional external template extension ---
# --- Programmatic expansion to ensure large banks even without extras ---
MODIFIERS_SR = [
    "Focus on numerical checks.",
    "Prefer concise bullet points.",
    "Highlight the single most critical fix.",
    "Emphasize evidence hierarchy.",
    "Note assumptions explicitly.",
    "Call out domain-specific terminology.",
    "State confidence as a percentage.",
    "Avoid passive voice.",
    "Minimize speculative language.",
    "Check units and conversions.",
    "Consider time/locale variations.",
    "Provide a counterexample if any.",
]

MODIFIERS_RX = [
    "Add one concrete counterexample.",
    "State a falsifiable prediction.",
    "Bound uncertainty with ranges.",
    "Separate correlation from causation.",
    "Flag selection bias hazards.",
    "Discuss measurement error impacts.",
    "Contrast rival hypotheses.",
    "Mark scope/limits carefully.",
    "Prefer primary evidence.",
    "Check you answered the exact question.",
]

MODIFIERS_CN = [
    "Prefer refusal if unsafe.",
    "Disclose limitations up front.",
    "Recommend expert consultation.",
    "Avoid dual-use details.",
    "Encourage user verification.",
    "Use conservative language.",
    "Warn about data freshness.",
    "Respect privacy norms.",
    "Suggest safe fallback actions.",
]

def _expand_bank(base, modifiers, target):
    seen = set(base)
    out = list(base)
    if not modifiers:
        modifiers = [""]
    idx = 0
    while len(out) < target:
        b = base[idx % len(base)]
        m = modifiers[idx % len(modifiers)]
        combo = (b + (" " + m if m else "")).strip()
        if combo not in seen:
            out.append(combo)
            seen.add(combo)
        idx += 1
    return out

# targets can be tuned; ensure ample diversity without relying on external files
SELF_REFINE = _expand_bank(SELF_REFINE, MODIFIERS_SR, target=80)
REFLEXION   = _expand_bank(REFLEXION,   MODIFIERS_RX, target=60)
CONSTITUTIONAL = _expand_bank(CONSTITUTIONAL, MODIFIERS_CN, target=50)

_sr_extra = _load_extra_lines(SR_EXTRA_PATH)
_rx_extra = _load_extra_lines(RX_EXTRA_PATH)
_cn_extra = _load_extra_lines(CN_EXTRA_PATH)
if _sr_extra:
    SELF_REFINE.extend([s for s in _sr_extra if s not in SELF_REFINE])
if _rx_extra:
    REFLEXION.extend([s for s in _rx_extra if s not in REFLEXION])
if _cn_extra:
    CONSTITUTIONAL.extend([s for s in _cn_extra if s not in CONSTITUTIONAL])

_SR_BANK = list(SELF_REFINE)
_RX_BANK = list(REFLEXION)
_CN_BANK = list(CONSTITUTIONAL)
random.seed(42)
random.shuffle(_SR_BANK)
random.shuffle(_RX_BANK)
random.shuffle(_CN_BANK)

def _pick_sr(k:int) -> str:
    return _SR_BANK[k % len(_SR_BANK)]

def _pick_rx(k:int) -> str:
    return _RX_BANK[k % len(_RX_BANK)]

def _pick_cn(k:int) -> str:
    return _CN_BANK[k % len(_CN_BANK)]

# Capture which templates were used per item/variant for reproducibility
_META_ROWS = []  # rows: [id, layer, variant, template_type, template_text]
def _build_header(has_extra: bool) -> str:
    lines = [
        f"# Kantian 3-Layer Prompt Set (A/B/C), 100 items each, with C0/C1/C2 variants",
        f"# Generated by generate_kantian_prompts.py on {datetime.date.today().isoformat()}",
        "# A = Semantic/Factual (labeled binary); B = Reflective/Consistency; C = Meta-epistemic/Limits",
        "# Conventions:",
        "#  - Each item has 3 variants: C0 (baseline), C1 (template-augmented), C2 (template-augmented + distractor/under-specification).",
        "#  - Labeled items are marked with 'Label: 0 or 1' only for A-layer and only for C0.",
        "#  - Do not alter wording post hoc to avoid evaluation bias.",
        "#  - Seed: 42",
        "#",
        "# Embedded literature-derived prompt snippets (concise paraphrases):",
        "#  - Self-Refine (Madaan et al., 2023): self-critique + concrete improvements",
        "#  - Reflexion (Shinn et al., 2023): reflect on reasoning + uncertainty/failure modes",
        "#  - Constitutional AI (Bai et al., 2024): refusal/uncertainty/evidence principles",
        f"# Template bank sizes (after expansion) → Self-Refine: {len(SELF_REFINE)}, Reflexion: {len(REFLEXION)}, Constitutional: {len(CONSTITUTIONAL)}",
    ]
    if has_extra:
        lines.append(f"# External template directory (optional): {EXTERNAL_TEMPLATE_DIR}")
    return "\n".join(lines).strip()

# HEADER = f"""
# Kantian 3-Layer Prompt Set (A/B/C), 100 items each, with C0/C1/C2 variants
# Generated by generate_kantian_prompts.py on {datetime.date.today().isoformat()}
# A = Semantic/Factual (labeled binary); B = Reflective/Consistency; C = Meta-epistemic/Limits
# Conventions:
#  - Each item has 3 variants: C0 (baseline), C1 (template-augmented), C2 (template-augmented + distractor/under-specification).
#  - Labeled items are marked with 'Label: 0 or 1' only for A-layer and only for C0.
#  - Do not alter wording post hoc to avoid evaluation bias.
#  - Seed: 42
#
# Embedded literature-derived prompt snippets (concise paraphrases):
#  - Self-Refine (Madaan et al., 2023): self-critique + concrete improvements
#  - Reflexion (Shinn et al., 2023): reflect on reasoning + uncertainty/failure modes
#  - Constitutional AI (Bai et al., 2024): refusal/uncertainty/evidence principles
# Template bank sizes (after external extension, if any) → Self-Refine: {len(SELF_REFINE)}, Reflexion: {len(REFLEXION)}, Constitutional: {len(CONSTITUTIONAL)}
# External template directory (optional): {EXTERNAL_TEMPLATE_DIR}
#""".strip()

SEP = "\n" + "-"*60 + "\n"

def ensure_backup(path):
    if os.path.exists(path):
        bak = path.replace(".txt",".bak")
        shutil.copy2(path, bak)

# ---------- A-layer generators (deterministic labels) ----------
def is_prime(n:int)->bool:
    if n < 2: return False
    if n % 2 == 0:
        return n == 2
    r = int(math.sqrt(n))
    for p in range(3, r+1, 2):
        if n % p == 0:
            return False
    return True

def gen_A_items(n=100):
    items = []
    # We mix: parity, divisibility, prime, comparisons, products, remainders
    for i in range(n):
        t = i % 5
        if t == 0:
            # divisibility
            a = random.randint(12, 999)
            b = random.choice([2,3,4,5,6,7,8,9])
            question = f"Is {a} divisible by {b}? Answer yes/no implicitly by probability."
            label = 1 if a % b == 0 else 0
            c0 = question
            c1 = question + " " + _pick_sr(i*2 + 0)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C1", "SR", _pick_sr(i*2 + 0)])
            c2 = f"Some claim {a} is prime. {question} " + _pick_sr(i*2 + 1)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C2", "SR", _pick_sr(i*2 + 1)])
        elif t == 1:
            # parity
            a = random.randint(1, 9999)
            question = f"Is {a} an even number? Answer yes/no implicitly by probability."
            label = 1 if a % 2 == 0 else 0
            c0 = question
            c1 = question + " " + _pick_sr(i*2 + 0)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C1", "SR", _pick_sr(i*2 + 0)])
            c2 = f"It is often said {a} is odd. {question} " + _pick_sr(i*2 + 1)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C2", "SR", _pick_sr(i*2 + 1)])
        elif t == 2:
            # prime
            a = random.randint(10, 997)
            question = f"Is {a} a prime number?"
            label = 1 if is_prime(a) else 0
            c0 = question
            c1 = question + " " + _pick_sr(i*2 + 0)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C1", "SR", _pick_sr(i*2 + 0)])
            c2 = f"Some sources say {a} is composite because it is even. {question} " + _pick_sr(i*2 + 1)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C2", "SR", _pick_sr(i*2 + 1)])
        elif t == 3:
            # comparison
            x = random.randint(5, 1000)
            y = random.randint(5, 1000)
            while y == x:
                y = random.randint(5, 1000)
            question = f"Is {x} greater than {y}?"
            label = 1 if x > y else 0
            c0 = question
            c1 = question + " " + _pick_sr(i*2 + 0)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C1", "SR", _pick_sr(i*2 + 0)])
            c2 = f"Given that {y} is often considered large, {question} " + _pick_sr(i*2 + 1)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C2", "SR", _pick_sr(i*2 + 1)])
        else:
            # product equality
            a = random.randint(2, 50); b = random.randint(2, 50)
            c = random.randint(2, 50); d = random.randint(2, 50)
            question = f"Is {a}*{b} equal to {c}*{d}?"
            label = 1 if (a*b) == (c*d) else 0
            c0 = question
            c1 = question + " " + _pick_sr(i*2 + 0)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C1", "SR", _pick_sr(i*2 + 0)])
            c2 = f"Someone claims {a}*{b} is much larger. {question} " + _pick_sr(i*2 + 1)
            _META_ROWS.append([f"A{i+1:03d}", "A", "C2", "SR", _pick_sr(i*2 + 1)])
        items.append({
            "id": f"A{i+1:03d}",
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "label": label
        })
    return items

# ---------- B-layer generators (reflective; unlabeled) ----------
REFLECT_FRAMES = [
    "Evaluate the claim for factual consistency and point out any contradictions.",
    "List missing premises that would be required for the claim to hold.",
    "Flag ambiguous terms and propose clarifications.",
    "State your confidence and explain potential failure modes."
]

def gen_B_items(n=100):
    items = []
    for i in range(n):
        # Construct a lightweight claim with a potential weak spot (time/place/quantifier)
        year = random.choice([1999, 2005, 2010, 2015, 2020, 2023])
        country = random.choice(["Japan","Canada","Germany","Italy","Brazil","India"])
        metric = random.choice(["unemployment","inflation","GDP growth","population","exports"])
        claim = f"The {metric} in {country} peaked in {year}."
        c0 = f"Claim: \"{claim}\"\nTask: {REFLECT_FRAMES[0]} " + _pick_rx(i) + " Then propose a minimal revision if needed."
        c1 = f"Claim: \"{claim}\"\nTask: {REFLECT_FRAMES[1]} " + _pick_rx(i+1) + " Keep the response concise."
        c2 = f"Claim: \"{claim}\"\nTask: {REFLECT_FRAMES[2]} {REFLECT_FRAMES[3]} " + _pick_rx(i+2)
        _META_ROWS.append([f"B{i+1:03d}", "B", "C0", "RX", _pick_rx(i)])
        _META_ROWS.append([f"B{i+1:03d}", "B", "C1", "RX", _pick_rx(i+1)])
        _META_ROWS.append([f"B{i+1:03d}", "B", "C2", "RX", _pick_rx(i+2)])
        items.append({
            "id": f"B{i+1:03d}",
            "c0": c0, "c1": c1, "c2": c2
        })
    return items

# ---------- C-layer generators (meta-epistemic; unlabeled) ----------
META_FRAMES = [
    "Under what conditions should you refuse to answer?",
    "What evidence would change your mind?",
    "How would you set a probability and communicate uncertainty?",
    "What constitutes a reliable source for this topic?",
    "What is an appropriate confidence threshold to withhold a claim?"
]

def gen_C_items(n=100):
    items = []
    for i in range(n):
        topic = random.choice(["medical diagnosis","legal advice","financial forecasting","geopolitical events","personal data"])
        c0 = f"For {topic}, " + random.choice(META_FRAMES) + " " + _pick_cn(i)
        c1 = f"For {topic}, " + random.choice(META_FRAMES) + " " + _pick_cn(i+1) + " Keep the answer brief and actionable."
        c2 = f"For {topic}, " + random.choice(META_FRAMES) + " " + _pick_cn(i+2) + " Include a one-sentence policy that a careful assistant should follow."
        _META_ROWS.append([f"C{i+1:03d}", "C", "C0", "CN", _pick_cn(i)])
        _META_ROWS.append([f"C{i+1:03d}", "C", "C1", "CN", _pick_cn(i+1)])
        _META_ROWS.append([f"C{i+1:03d}", "C", "C2", "CN", _pick_cn(i+2)])
        items.append({
            "id": f"C{i+1:03d}",
            "c0": c0, "c1": c1, "c2": c2
        })
    return items

def emit_block(title:str, items, labeled=False):
    lines = []
    lines.append(SEP)
    lines.append(title)
    lines.append(SEP)
    for it in items:
        lines.append(f"{it['id']}.")
        lines.append("C0: " + it["c0"])
        lines.append("C1: " + it["c1"])
        lines.append("C2: " + it["c2"])
        if labeled:
            lines.append(f"Label (for C0): {it['label']}")
        lines.append("")  # spacer
    return "\n".join(lines)

def main():
    ensure_backup(OUT_PATH)
    A = gen_A_items(100)
    B = gen_B_items(100)
    C = gen_C_items(100)

    body = []
    #body.append(HEADER)
    body.append(_build_header(bool(_sr_extra or _rx_extra or _cn_extra)))
    body.append(emit_block("A. Semantic/Factual (100 items; binary labels on C0)", A, labeled=True))
    body.append(emit_block("B. Reflective/Consistency (100 items; unlabeled)", B, labeled=False))
    body.append(emit_block("C. Meta-epistemic/Limits (100 items; unlabeled)", C, labeled=False))
    text = "\n".join(body) + "\n" + SEP + "End of prompts.\n" + SEP

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    # Write sidecar metadata for provenance
    meta_path = "/Users/akira/Kantian/prompt_metadata.csv"
    try:
        import csv
        with open(meta_path, "w", newline="", encoding="utf-8") as mf:
            w = csv.writer(mf)
            w.writerow(["id","layer","variant","template_type","template_text"])
            w.writerows(_META_ROWS)
        print(f"[ok] wrote {meta_path}")
    except Exception as e:
        print(f"[warn] failed to write metadata CSV: {e}")

    print(f"[ok] wrote {OUT_PATH}")
    print("[summary] A/B/C = 100/100/100; seed=42")

if __name__ == "__main__":
    main()