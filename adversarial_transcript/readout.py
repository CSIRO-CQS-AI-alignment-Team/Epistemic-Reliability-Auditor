"""Dual-order forced-choice readout for transcript-conditioned analysis.

For a semantic hypothesis ``h`` shown at A in one order and B in the reverse order:

    l_A = logodds(h | h at A)
    l_B = logodds(h | h at B)
    m = (l_A + l_B) / 2
    l_pos = (l_B - l_A) / 2
    p_sem = sigmoid(m)

``m`` and ``p_sem`` are the order-combined semantic readout; ``l_pos`` records answer-
position bias separately. The helper produces transcript-conditioned ``H_false`` and
``Y_true`` margins plus prompt-matched no-transcript controls used by the selected-
transcript MI analysis.

Scored prompts are cached by checkpoint identity, option seed, template identity, and
exact prompt hash. The append-only JSONL cache writes and fsyncs one complete entry at a
time for resumable execution.
"""

import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# score_verifier's module-level imports are stdlib + quote_utils (torch is lazy inside
# ForcedChoiceVerifier), so importing it here keeps the CPU/offline path clean.
from adversarial_transcript import fingerprint as fingerprint_mod  # noqa: E402
from adversarial_transcript.score_verifier import (  # noqa: E402
    DEFAULT_OPTION_SEED,
    PROMPT_VERSION as SCORE_VERIFIER_PROMPT_VERSION,
    QH_READOUT_TEMPLATE,
    QY_CLEAN_READOUT_TEMPLATE,
    QY_READOUT_TEMPLATE,
    _answer_text,
    _debater_for_stance,
    deterministic_answer_order,
    render_verified_transcript,
)

# Stable protocol identity recorded in readout caches and result artifacts.
READOUT_VERSION = "adv-prompt-opt-readout-v1"

# Neutral transcript block for the prompt-matched T0 baseline used by MI analysis.
NO_TRANSCRIPT_MARKER = "(no transcript provided)"

HYPOTHESIS = {"Q_H": ("H_false", "H_true"), "Q_Y": ("Y_true", "Y_false")}


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TEMPLATE_SET_ID = "score-verifier-production"
TEMPLATE_SET_SHA256 = _sha256_text("|".join([
    SCORE_VERIFIER_PROMPT_VERSION, QH_READOUT_TEMPLATE, QY_READOUT_TEMPLATE,
    QY_CLEAN_READOUT_TEMPLATE, NO_TRANSCRIPT_MARKER,
]))


def prompt_hash(prompt):
    return _sha256_text(prompt)


def sigmoid(x):
    """Numerically stable logistic. math.exp(-x) overflows for x <~ -709."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ---- order helpers --------------------------------------------------------
def orders_for(candidate, q_key, option_seed=DEFAULT_OPTION_SEED):
    """The two semantic answer orders (canonical + swap) as {'AB': .., 'BA': ..}.

    Both are always scored and the margin averages them, so which order is LABELLED
    'AB' cannot affect m or l_pos -- it only fixes the reporting order.
    """
    canonical = deterministic_answer_order(candidate, q_key, option_seed)
    return {"AB": canonical, "BA": {"A": canonical["B"], "B": canonical["A"]}}


def semantic_logodds(letter_logprobs, order, hypothesis):
    """(log-odds of `hypothesis`, the letter it was shown at) for one order."""
    hyp_letter = next(letter for letter, sem in order.items() if sem == hypothesis)
    other_letter = "B" if hyp_letter == "A" else "A"
    return letter_logprobs[hyp_letter] - letter_logprobs[other_letter], hyp_letter


def decompose(order_blocks, hypothesis):
    """Reduce both scored orders to the semantic margin + position bias.

    `l_A`/`l_B` are keyed by WHERE the hypothesis sat (letter A or letter B), not by
    which block was labelled AB, so the decomposition is label-invariant.
    """
    by_letter = {}
    per_order = {}
    for tag, block in order_blocks.items():
        logodds, hyp_letter = semantic_logodds(
            block["letter_logprobs"], block["order"], hypothesis)
        by_letter[hyp_letter] = logodds
        per_order[tag] = {
            "order": dict(block["order"]),
            "letter_logprobs": dict(block["letter_logprobs"]),
            "logodds": logodds,
            "hypothesis_letter": hyp_letter,
            "prompt_sha256": block.get("prompt_sha256"),
        }
    if set(by_letter) != {"A", "B"}:
        raise ValueError("both answer orders (hypothesis at A and at B) are required")
    l_A, l_B = by_letter["A"], by_letter["B"]
    margin = 0.5 * (l_A + l_B)
    return {
        "hypothesis": hypothesis,
        "orders": per_order,
        "l_A": l_A,
        "l_B": l_B,
        "m": margin,
        "l_pos": 0.5 * (l_B - l_A),
        "p_sem": sigmoid(margin),
        # Per-orientation decisions. The Q_Y hard gate requires BOTH to favour the
        # hypothesis, so a candidate cannot pass by averaging two opposite
        # position-biased outcomes into a positive mean.
        "decision_by_orientation": {"hypothesis_at_A": l_A > 0.0,
                                    "hypothesis_at_B": l_B > 0.0},
        "decision_all_orientations": (l_A > 0.0) and (l_B > 0.0),
        "decision_semantic": margin > 0.0,
        "prompt_sha256_by_order": {tag: block.get("prompt_sha256")
                                   for tag, block in order_blocks.items()},
    }


# ---- prompt builders (reuse the production templates verbatim) ------------
def qh_prompt(candidate, order, transcript_text):
    return QH_READOUT_TEMPLATE.format(
        visible_question=candidate["qy"]["question"],
        debater_a_answer=_answer_text(candidate, "Q_Y", candidate["transcript"]["Debater A"]),
        debater_b_answer=_answer_text(candidate, "Q_Y", candidate["transcript"]["Debater B"]),
        transcript=transcript_text,
        question=candidate["qh"]["question"],
        answer_a=_answer_text(candidate, "Q_H", order["A"]),
        answer_b=_answer_text(candidate, "Q_H", order["B"]),
    )


def qy_conditioned_prompt(candidate, order, transcript_text):
    return QY_READOUT_TEMPLATE.format(
        question=candidate["qy"]["question"],
        answer_a=_answer_text(candidate, "Q_Y", order["A"]),
        answer_b=_answer_text(candidate, "Q_Y", order["B"]),
        name_a=_debater_for_stance(candidate["transcript"], order["A"]),
        name_b=_debater_for_stance(candidate["transcript"], order["B"]),
        transcript=transcript_text,
    )


def qy_clean_prompt(candidate, order):
    return QY_CLEAN_READOUT_TEMPLATE.format(
        question=candidate["qy"]["question"],
        answer_a=_answer_text(candidate, "Q_Y", order["A"]),
        answer_b=_answer_text(candidate, "Q_Y", order["B"]),
    )


# ---- cache ----------------------------------------------------------------
class ReadoutCache:
    """Append-only JSONL cache of scored letter log-probabilities."""

    def __init__(self, path=None):
        self.path = path
        self._entries = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line from an interrupted run: everything before it
                        # is still valid, so drop it rather than losing the whole cache.
                        continue
                    self._entries[row["key"]] = row["letter_logprobs"]

    @staticmethod
    def key(scorer_model, option_seed, template_set, prompt_sha256):
        return f"{scorer_model}|{option_seed}|{template_set}|{prompt_sha256}"

    def get(self, key):
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(value)

    def put(self, key, letter_logprobs):
        self._entries[key] = dict(letter_logprobs)
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "letter_logprobs": letter_logprobs},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def stats(self):
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}


def dry_run_logprobs(key):
    """Deterministic pseudo log-probabilities so the whole pipeline runs on CPU."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0x100000000
    logodds = (fraction - 0.5) * 4.0
    p_a = sigmoid(logodds)
    return {"A": math.log(p_a), "B": math.log(1.0 - p_a)}


def release_scorer(scorer, log=None):
    """Free a loaded verifier's GPU memory now, not whenever the GC runs.

    Duck-typed so a fake scorer, a ForcedChoiceVerifier, or any future wrapper works.
    """
    if scorer is None:
        return False
    released = False
    releaser = getattr(scorer, "release", None)
    if callable(releaser):
        releaser()
        released = True
    else:
        # Assign None rather than delattr: the heavy reference may live on the CLASS
        # (or be a property), where delattr raises and would leave the weights resident.
        for attribute in ("model", "tokenizer", "_choice_logprob_cache"):
            if hasattr(scorer, attribute):
                try:
                    setattr(scorer, attribute, None)
                    released = True
                except (AttributeError, TypeError):
                    pass
    import gc  # noqa: PLC0415
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - freeing memory must never fail a run
            pass
    if log:
        log(f"[readout] released scorer {getattr(scorer, 'model_name', scorer)!r}")
    return released


class Reader:
    """Scores prompts through a ForcedChoiceVerifier (or deterministically in dry-run).

    The cache key is bound to the checkpoint's CONTENT fingerprint, not its path: training
    overwrites checkpoints in place, so a path-keyed cache would serve yesterday's logits
    for today's weights. `release()` gives callers an explicit lifecycle so two 20B models
    are never resident at once.
    """

    def __init__(self, scorer=None, model_label="dry-run", option_seed=DEFAULT_OPTION_SEED,
                 cache=None, dry_run=False, template_set=TEMPLATE_SET_ID,
                 checkpoint_fingerprint=None):
        if not dry_run and scorer is None:
            raise ValueError("a real Reader needs a scorer; pass dry_run=True otherwise")
        if not dry_run and checkpoint_fingerprint is None:
            raise ValueError(
                "a real Reader needs a checkpoint fingerprint: a path-only identity would "
                "let an in-place weight overwrite silently reuse stale cached logits")
        self.scorer = scorer
        self.model_label = model_label
        self.option_seed = option_seed
        self.cache = cache if cache is not None else ReadoutCache()
        self.dry_run = dry_run
        self.template_set = template_set
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.calls = 0
        self.released = False

    def release(self, log=None):
        """Free the scorer's GPU memory and make further scoring impossible."""
        if self.released:
            return False
        released = release_scorer(self.scorer, log=log)
        self.scorer = None
        self.released = True
        return released

    def scorer_id(self):
        """The CONTENT identity used in cache keys. Never a bare path for a real run."""
        if self.dry_run:
            return fingerprint_mod.identity(
                self.checkpoint_fingerprint
                or fingerprint_mod.dry_run_fingerprint(self.model_label))
        return fingerprint_mod.identity(self.checkpoint_fingerprint)

    def letter_logprobs(self, prompt):
        if self.released:
            raise RuntimeError(
                "this Reader's scorer has been released; a later scoring call would need "
                "to reload the model. Build a fresh Reader instead of resurrecting one.")
        sha = prompt_hash(prompt)
        key = ReadoutCache.key(self.scorer_id(), self.option_seed, self.template_set, sha)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, sha
        value = dry_run_logprobs(key) if self.dry_run else self.scorer.choice_logprobs(prompt)
        value = {letter: float(value[letter]) for letter in ("A", "B")}
        self.calls += 1
        self.cache.put(key, value)
        return value, sha

    def block(self, candidate, q_key, prompt_fn, hypothesis=None):
        hypothesis = hypothesis or HYPOTHESIS[q_key][0]
        order_blocks = {}
        for tag, order in orders_for(candidate, q_key, self.option_seed).items():
            prompt = prompt_fn(order)
            logprobs, sha = self.letter_logprobs(prompt)
            order_blocks[tag] = {"order": order, "letter_logprobs": logprobs,
                                 "prompt_sha256": sha}
        out = decompose(order_blocks, hypothesis)
        out["max_letter_logprob"] = max(
            max(block["letter_logprobs"].values()) for block in order_blocks.values())
        return out

    def provenance(self):
        return {
            "readout_version": READOUT_VERSION,
            "readout": "final_channel_answer_slot_both_orders",
            "scorer_model": self.model_label,
            "scorer_identity": self.scorer_id(),
            "scorer_checkpoint": fingerprint_mod.describe(self.checkpoint_fingerprint),
            "scorer_model_quantized_load": (
                None if (self.dry_run or self.scorer is None)
                else getattr(self.scorer, "quantized_load", None)),
            "option_seed": self.option_seed,
            "template_set": self.template_set,
            "template_set_sha256": TEMPLATE_SET_SHA256,
            "score_verifier_prompt_version": SCORE_VERIFIER_PROMPT_VERSION,
            "both_orders": True,
            "dry_run": self.dry_run,
        }


# ---- composite readouts ---------------------------------------------------
def read_conditioned(reader, candidate, story):
    """Q_H and Q_Y margins given a candidate transcript (rendered once, reused)."""
    rendered = render_verified_transcript(candidate, story)
    return {
        "qh": reader.block(candidate, "Q_H", lambda o: qh_prompt(candidate, o, rendered)),
        "qy": reader.block(candidate, "Q_Y",
                           lambda o: qy_conditioned_prompt(candidate, o, rendered)),
    }


def read_matched_t0(reader, candidate):
    """Prompt-matched no-transcript baseline: same framing, neutral transcript block.

    Q_H keeps the full visible-question/debater-answer framing so ONLY the transcript
    content differs; Q_Y uses the production clean template. No story is needed.
    """
    return {
        "qh": reader.block(candidate, "Q_H",
                           lambda o: qh_prompt(candidate, o, NO_TRANSCRIPT_MARKER)),
        "qy_clean": reader.block(candidate, "Q_Y",
                                 lambda o: qy_clean_prompt(candidate, o)),
    }


def read_honest_control(reader, honest_candidate, story):
    """Matched honest control: the item's honest transcript, scored identically."""
    return read_conditioned(reader, honest_candidate, story)
