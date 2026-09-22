"""Estimate feature-posterior coupling for one selected-transcript condition.

Each invocation computes

    r_post,c^i = p_sem(V_c, H_false | Q_H^i, T_c^i)
                 - p_sem(V_base, H_false | Q_H^i, T_c^i)
    MI_post,c = KSG(F_c, r_post,c)

for ``c`` selected by ``--mode {honest,adversarial}``. One observation is one final
selected best-of-eight transcript. Honest and adversarial runs use the strict common
support of structurally valid transcript pairs: 184 QuALITY-H items and 82 GPQA items in
the released analysis.

``F_c`` contains six coordinates in fixed order: target alignment, semantic-role length
imbalance, verified-evidence asymmetry, semantic escalation, uncertainty suppression,
and hidden-task relevance. Semantic coordinates use the pinned all-MiniLM-L6-v2
encoder; the story is used only for offline quote verification and is never embedded or
included in a verifier prompt.

Both checkpoints score the same transcript and hidden forced-choice prompt in both A/B
orders at the Harmony final-channel answer slot. The two semantic log-odds are averaged
before applying the sigmoid to obtain ``p_sem``. Checkpoint fingerprints, prompt hashes,
and readout identities prevent incompatible caches or model contents from being mixed.

The KSG estimate is a finite-sample cross-pair dependence measure in nats. It is
conditioned on best-of-eight winner selection, does not control every pair-specific
difficulty, and is not a causal effect or a direct estimate of worst-case steering
capacity. Point estimates alone do not establish statistical significance.

Examples:
    python3 mi-post-adv-honest.py --mode adversarial --dataset QuALITY-H \
        --devices 0,1,2,3 --output-dir runs/mi-post/QuALITY-H/adversarial
    python3 mi-post-adv-honest.py --mode honest --dataset GPQA \
        --devices 0,1,2,3 --output-dir runs/mi-post/GPQA/honest
"""

import argparse
import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import os
import queue
import re
import sys
import threading
from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _HERE
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Dependency direction is downstream MI analysis -> adversarial_transcript, never the
# reverse. Every module
# below is stdlib-only at import time (torch is lazy inside ForcedChoiceVerifier), so this
# file stays importable on a CPU-only, network-free machine.
from adversarial_transcript import common as at_common                      # noqa: E402
from adversarial_transcript import fingerprint as fingerprint_mod           # noqa: E402
from adversarial_transcript.readout import (                                # noqa: E402
    READOUT_VERSION as ADVERSARIAL_TRANSCRIPT_READOUT_VERSION,
    Reader,
    ReadoutCache,
    TEMPLATE_SET_ID,
    TEMPLATE_SET_SHA256,
    orders_for,
    prompt_hash,
    qh_prompt,
    release_scorer,
)

# Reader.block() applies readout.decompose(order_blocks, "H_false") to the two scored orders
# and, under dry_run, sources its letter log-probabilities from readout.dry_run_logprobs.
# Both are reached through Reader rather than re-implemented here; the hypothesis is passed
# EXPLICITLY at every call site (never left to an upstream default) and asserted per row.
from adversarial_transcript.score_verifier import (                         # noqa: E402
    DEFAULT_OPTION_SEED,
    PROMPT_VERSION as SCORE_VERIFIER_PROMPT_VERSION,
    QH_READOUT_TEMPLATE,
    render_verified_transcript,
)
from quote_utils import normalize_text, verify_quotes                       # noqa: E402

# ---- identity strings ------------------------------------------------------
READOUT_ID = "mi-post-readout-forced-choice-dual-order-v1"
# The hypothesis whose posterior IS the response. Passed explicitly at every scoring call.
HYPOTHESIS = "H_false"
READOUT_FAMILY = "final_channel_answer_slot_forced_choice_no_cot"
# Readout families that must never be pooled with this one (post-CoT vLLM generated-slot).
FORBIDDEN_READOUT_PREFIXES = ("base-transcript-eval",)
FEATURE_VERSION = "mi-post-features-v1"
ESTIMATOR_VERSION = "mi-post-ksg1-mode-robust-scaler-v2"
OBSERVATIONS_SCHEMA_VERSION = "mi-post-observations-v2"
RESULTS_SCHEMA_VERSION = "mi-post-results-v2"
# v2 adds dataset+mode to the sidecar identity. v1 sidecars did not record which
# dataset/mode filled a cache, so they are ambiguous and are refused rather than
# upgraded in place: re-stamping one would launder an unknown population into a run.
CACHE_META_SCHEMA_VERSION = "mi-post-readout-cache-meta-v2"
READOUT_EXECUTION_VERSION = "mi-post-resident-thread-pool-v2"

CONDITIONS = ("honest", "adv")
TRANSCRIPT_FIELDS = {"honest": "honest_transcript", "adv": "adversarial_transcript"}
MODE_CHOICES = ("adversarial", "honest")
MODE_TO_CONDITION = {"adversarial": "adv", "honest": "honest"}
CHECKPOINT_ROLES = ("pre", "post")

# ---- feature contract ------------------------------------------------------
# Frozen six-coordinate vector in report order. All six enter the joint KSG estimand.
FEATURE_NAMES = (
    "semantic_target_alignment",
    "len_imbalance_ytrue",
    "verified_evidence_density_asymmetry",
    "semantic_escalation",
    "uncertainty_suppression",
    "hidden_task_relevance",
)

HYPOTHESIS_ANCHOR_TEMPLATE = "Question: {question}\nProposed answer: {answer}"
ANCHOR_TEMPLATE_VERSION = "question-proposed-answer-v1"

SENTENCE_SPLITTER_VERSION = "regex-terminal-punct-v1"
# Split at the whitespace after . ! ? plus any trailing closing quote/bracket. The closer is
# CAPTURED and handed back to the sentence it belongs to, so splitting never eats a character.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])([\"'’”)\]]*)\s+")

# Chunking uses the encoder's token budget, not a whitespace-word proxy. The usable
# budget subtracts tokenizer-added special tokens; every chunk is verified with the real
# tokenizer, and over-budget text is rejected before encoding.
CHUNKER_VERSION = "tokenizer-aware-greedy-verify-v1"
# The synthetic budget/tokenizer used by the explicitly-fake encoder (dry-run + tests): one
# "token" per whitespace word, same nominal budget, so the chunking path is exercised on CPU
# without downloading MiniLM.
FAKE_ENCODER_TOKEN_BUDGET = 254

ANCHOR_KEYS = ("a_hfalse", "a_htrue", "e_qh", "e_qy")

# Fixed, versioned hedging/uncertainty lexicon. Case-insensitive, whole word / whole
# declared phrase only. Longest alternative wins at a position, so "it seems" is one match
# rather than two and "may" never matches inside "maybe".
UNCERTAINTY_LEXICON_VERSION = "uncertainty-hedging-v1"
UNCERTAINTY_LEXICON = (
    "allegedly", "ambiguous", "apparently", "appear", "appears", "arguably",
    "assume", "assumes", "assuming", "conceivably", "could", "difficult to say",
    "guess", "hard to say", "i believe", "i suspect", "i think", "in my opinion",
    "in my view", "it appears", "it is possible", "it is unclear", "it seems",
    "it's possible", "it's unclear", "kind of", "likely", "may", "maybe", "might",
    "more or less", "not certain", "not sure", "perhaps", "plausibly", "possibility",
    "possible", "possibly", "potentially", "presumably", "presume", "probably",
    "roughly", "seem", "seemed", "seemingly", "seems", "somewhat", "sort of",
    "speculate", "speculation", "suggest", "suggesting", "suggests", "tend to",
    "tends to", "to some extent", "uncertain", "unclear", "unlikely",
)

# ---- embedding contract ----------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
EMBEDDING_IMPL_VERSION = "mi-post-embed-l2-mean-pool-v1"
FAKE_ENCODER_NAME = "deterministic-fake-encoder-v1"
FAKE_ENCODER_DIM = 32

# ---- dequantization contract (vendored, see below) -------------------------
DEQUANT_N_SEEDS = 5
DEQUANT_AMPLITUDE = 0.5
DEQUANT_AGG = "median"
# The repo's single deterministic seed constant (adversarial_transcript.common.STANCE_SEED
# == score_verifier.DEFAULT_OPTION_SEED). Reused as the dequantization global seed so the
# jitter is byte-reproducible across processes and machines.
GLOBAL_SEED = at_common.STANCE_SEED

# ---- residue / eligibility -------------------------------------------------
# Mirrors verifier-base-transcript-eval.py:RESIDUE_MARKERS. Flagged rows are KEPT: the
# deployed transcripts are the estimand.
RESIDUE_MARKERS = ("<thinking>", "<TRUNCATED>")
SKIP_TRANSCRIPT_NULL = "transcript_null"
SKIP_STANCE_MALFORMED = "stance_malformed"
SKIP_ROUNDS_MISSING = "rounds_missing_or_empty"
SKIP_EMPTY_ROUND = "empty_round_argument"
SKIP_STORY_MISSING = "story_missing"
SKIP_NONFINITE_FEATURES = "nonfinite_features"
SKIP_READOUT_INVALID = "readout_invalid"
SKIP_NOT_IN_COMMON_SUPPORT = "not_in_common_support"
SKIP_REASONS = (SKIP_TRANSCRIPT_NULL, SKIP_STANCE_MALFORMED, SKIP_ROUNDS_MISSING,
                SKIP_EMPTY_ROUND, SKIP_STORY_MISSING, SKIP_NONFINITE_FEATURES,
                SKIP_READOUT_INVALID, SKIP_NOT_IN_COMMON_SUPPORT)

# ---- dataset / checkpoint contract -----------------------------------------
DATASET_CHOICES = ("QuALITY-H", "GPQA")
DEFAULT_DATASET = "QuALITY-H"
PRE_CHECKPOINT_RELATIVE = "checkpoints/gpt-oss-20b-bf16-base"
POST_CHECKPOINT_TEMPLATES = {
    "adversarial": "checkpoints/gpt-oss-20b-verifier-fullft-adversarial-%s",
    "honest": "checkpoints/gpt-oss-20b-verifier-fullft-%s",
}
TRAINING_METADATA_FILE = "training-metadata.json"

DEFAULT_K = 5
# Health flag threshold, mirrors verifier-posterior-eval.py / score_verifier readout health.
SUSPECT_MAX_LETTER_LOGPROB = -8.0
SATURATION_EPS = 1e-6

# Hugging Face / Accelerate checkpoint construction temporarily mutates process-global
# materialisation hooks.  Four replicas are therefore constructed serially, even though
# all item-level forwards run concurrently after every requested replica is resident.
_HF_LOAD_LOCK = threading.RLock()


class MiPostError(RuntimeError):
    """Any refusal that must stop the run rather than degrade it silently."""


class BudgetExhausted(MiPostError):
    """A wrapped chunk's fixed overhead already fills the encoder's token budget.

    Raised only when a hypothesis ANSWER alone leaves no room for any question content, so
    "the same proposed answer in every chunk" is impossible. The row is then failed (its
    embedding coordinates become non-finite and the eligibility ledger records why) rather
    than quietly embedding a truncated or answer-free anchor.
    """


# ===========================================================================
# Deterministic identity, feature, and KSG helpers
# ===========================================================================

# ---- deterministic identities ---------------------------------------------
def stable_seed(*parts):
    """Deterministic 64-bit seed from stringified key parts joined with '|'.

    Python's built-in hash() is salted per process, so it must never seed anything here.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def content_hash(obj):
    """Return the canonical SHA-256 of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- transcript feature helpers -------------------------------------------
_QUOTE_TAG = re.compile(r"</?(?:v_|u_)?quote>", re.IGNORECASE)


def _wordcount(text):
    """Words after removing quote MARKUP but keeping the quoted words themselves."""
    return len(_QUOTE_TAG.sub(" ", text).split())


def ytrue_defender(transcript):
    """The debater NAME defending Y_true. Semantic stance role, never a fixed position."""
    matches = [n for n in at_common.DEBATER_NAMES if transcript.get(n) == "Y_true"]
    stances = {transcript.get(n) for n in at_common.DEBATER_NAMES}
    if len(matches) != 1 or stances != set(at_common.STANCE_LABELS):
        raise MiPostError("malformed stance map: need exactly one Y_true / one Y_false defender")
    return matches[0]


def yfalse_defender(transcript):
    ytrue = ytrue_defender(transcript)
    return (at_common.DEBATER_NAMES[0] if ytrue == at_common.DEBATER_NAMES[1]
            else at_common.DEBATER_NAMES[1])


def round_word_counts(transcript):
    """Per-round (words_by_Ytrue_defender, words_by_Yfalse_defender) as integers."""
    ytrue = ytrue_defender(transcript)
    yfalse = yfalse_defender(transcript)
    return [(_wordcount(rnd[ytrue]), _wordcount(rnd[yfalse]))
            for rnd in transcript.get("rounds", [])]


def _len_imbalance_from_counts(counts):
    """(sum_Ytrue - sum_Yfalse) / total, over per-round (a, b) counts (ints or floats)."""
    if not counts:
        return float("nan")
    tot_true = float(sum(c[0] for c in counts))
    tot_false = float(sum(c[1] for c in counts))
    total = tot_true + tot_false
    if abs(total) <= 1e-9:
        return float("nan")
    return (tot_true - tot_false) / total


def len_imbalance_raw(transcript):
    """Undequantized length imbalance (diagnostic / test oracle)."""
    return _len_imbalance_from_counts(round_word_counts(transcript))


def len_imbalance_dequant(transcript, key, global_seed=GLOBAL_SEED,
                          n_seeds=DEQUANT_N_SEEDS, amplitude=DEQUANT_AMPLITUDE,
                          agg=DEQUANT_AGG):
    """Dequantized length imbalance: jitter the INTEGER per-round word counts by
    Uniform(-amp, amp) over `n_seeds` predeclared seeds and take the median.

    Word counts are integers, so the raw ratio is lattice-valued and ties break the KSG
    continuity assumption. The jitter is always applied and never a post-hoc choice; `key`
    (the transcript content hash) makes it deterministic per transcript. The two uniform
    draws per round follow a fixed round-major RNG stream, making the dequantized value
    deterministic for each transcript.
    """
    base = round_word_counts(transcript)
    values = []
    for s in range(n_seeds):
        rng = np.random.default_rng(stable_seed(global_seed, "dequant", key, s))
        jittered = [(a + rng.uniform(-amplitude, amplitude),
                     b + rng.uniform(-amplitude, amplitude)) for (a, b) in base]
        values.append(_len_imbalance_from_counts(jittered))
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    reducer = np.nanmedian if agg == "median" else np.nanmean
    return float(reducer(arr))


def ols_slope(values):
    """Return the OLS slope of ``values`` against the index t = 1..R.

    Fewer than two points, or a degenerate design, gives NaN rather than a fabricated 0.
    """
    xs = np.asarray(values, dtype=float)
    if xs.size < 2 or not np.isfinite(xs).all():
        return float("nan")
    t = np.arange(1.0, xs.size + 1.0)
    tbar = t.mean()
    denom = float(((t - tbar) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(((t - tbar) * (xs - xs.mean())).sum() / denom)


# ---- KSG1 estimator ---------------------------------------------------------
def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _ksg_validate(X, Y, k):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    n = X.shape[0]
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("KSG input contains NaN/Inf")
    if not (isinstance(k, (int, np.integer)) and k >= 1):
        raise ValueError("k must be a positive integer")
    if k >= n:
        raise ValueError(f"need k < N (got k={k}, N={n})")
    for name, A in (("X", X), ("Y", Y)):
        if np.ptp(A, axis=0).max() == 0.0:
            raise ValueError(f"{name} has a zero-variance (constant) coordinate")
    return n


def ksg_mi(X, Y, k=DEFAULT_K):
    """KSG1 estimate of MI(X;Y) in NATS.

    MI = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >, with the k-th nearest neighbour
    distance taken in the JOINT space under the Chebyshev (max) norm and n_x / n_y counting
    points STRICTLY within that distance in each marginal. Estimates in this family can be
    negative near MI = 0; they are returned RAW and never clipped.
    """
    X, Y = _as2d(X), _as2d(Y)
    n = _ksg_validate(X, Y, k)
    Z = np.hstack([X, Y])
    eps = cKDTree(Z).query(Z, k + 1, p=np.inf)[0][:, k]
    r = np.nextafter(eps, 0.0)          # strict "< eps" via the largest float below eps
    nx = np.asarray(cKDTree(X).query_ball_point(X, r, p=np.inf, return_length=True)) - 1
    ny = np.asarray(cKDTree(Y).query_ball_point(Y, r, p=np.inf, return_length=True)) - 1
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def _ksg_mi_brute(X, Y, k=DEFAULT_K):
    """Pure-numpy O(n^2) Chebyshev oracle. TEST ONLY -- production always uses ksg_mi."""
    X, Y = _as2d(X), _as2d(Y)
    n = _ksg_validate(X, Y, k)
    dx = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
    dy = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
    dz = np.maximum(dx, dy)
    eps = np.sort(dz, axis=1)[:, k]
    nx = np.maximum((dx < eps[:, None]).sum(axis=1) - 1, 0)
    ny = np.maximum((dy < eps[:, None]).sum(axis=1) - 1, 0)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def gaussian_mi(rho):
    """True MI (nats) of a bivariate standard normal with correlation rho. TEST ORACLE."""
    return -0.5 * float(np.log(1.0 - rho ** 2))


def rho_for_mi(target_mi):
    """Correlation giving a bivariate-Gaussian MI of `target_mi` nats. TEST ORACLE."""
    return float(np.sqrt(1.0 - np.exp(-2.0 * target_mi)))


# ---- robust feature scaling ------------------------------------------------
def robust_scale_params(M):
    """Robust per-column scaler: centre = median, scale = 1.4826 * MAD;
    MAD = 0 and var > 0 -> std; both zero -> the coordinate is CONSTANT and is dropped."""
    M = np.asarray(M, float)
    med = np.median(M, axis=0)
    mad = np.median(np.abs(M - med), axis=0) * 1.4826
    std = M.std(axis=0)
    scale = np.where(mad > 0, mad, np.where(std > 0, std, 1.0))
    keep = (mad > 0) | (std > 0)
    return med, scale, keep


def apply_scale(M, params):
    med, scale, keep = params
    Z = (np.asarray(M, float) - med) / scale
    return Z[:, keep]


# ===========================================================================
# Text extractors (versioned; all operate on the REVERIFIED per-turn strings)
# ===========================================================================
def strip_quote_markup(text):
    """Drop <quote>/<v_quote>/<u_quote> MARKUP, keep the quoted words."""
    return _QUOTE_TAG.sub(" ", text)


def normalise_whitespace(text):
    return " ".join(text.split())


def split_sentences(text):
    """Deterministic, dependency-free sentence split (SENTENCE_SPLITTER_VERSION).

    Splits at whitespace following terminal punctuation (with any trailing closing quote or
    bracket). Text with no terminal punctuation stays one sentence; nothing is dropped.
    """
    flat = normalise_whitespace(text)
    if not flat:
        return []
    out, start = [], 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(flat):
        end = match.start() + len(match.group(1))     # keep the closing punctuation
        out.append(flat[start:end])
        start = match.end()
    out.append(flat[start:])
    return [s for s in (part.strip() for part in out) if s]


class Chunker:
    """Splits text so that every emitted string fits the encoder's real token budget.

    `count_tokens` is the encoder's own tokenizer (CONTENT tokens, no specials) and `budget`
    is its usable window after reserving special-token capacity. Nothing is ever truncated:
    a long text becomes several chunks and the caller pools them.

    `wrap` lets a chunk be measured INSIDE a template. Target anchors use it so that every
    emitted chunk is a complete "Question: <part>\nProposed answer: <answer>" string -- the
    proposed answer is repeated in each chunk, which is what keeps the H_true/H_false contrast
    present in every embedded piece instead of only in the first one.

    The algorithm is estimate-then-verify: per-word token counts give a fast greedy candidate,
    then the ASSEMBLED string is measured with the real tokenizer and shrunk until it fits, so
    sub-word merging across a join can never push a chunk over the limit.
    """

    def __init__(self, count_tokens, budget, version=CHUNKER_VERSION):
        if budget is None or int(budget) < 1:
            raise MiPostError("chunker needs a positive token budget, got %r" % (budget,))
        self.budget = int(budget)
        self.version = version
        self._count = count_tokens
        self._memo = {}
        self.calls = 0

    def count(self, text):
        """Memoised token count. The same strings are measured by staging and by the gate."""
        cached = self._memo.get(text)
        if cached is None:
            cached = int(self._count(text))
            self._memo[text] = cached
            self.calls += 1
        return cached

    def fits(self, text):
        return self.count(text) <= self.budget

    def chunk(self, text, wrap=None):
        """-> list of ready-to-encode strings, each within budget, covering all of `text`."""
        emit = wrap if wrap is not None else (lambda s: s)
        measure = (lambda s: self.count(emit(s)))
        flat = normalise_whitespace(text)
        if not flat and wrap is None:
            return []
        if measure(flat) <= self.budget:
            return [emit(flat)]
        overhead = measure("")
        if overhead >= self.budget:
            raise BudgetExhausted(
                "wrapper overhead is %d tokens with a budget of %d: no question content can "
                "accompany this answer" % (overhead, self.budget))
        inner_budget = self.budget - overhead
        words = flat.split()
        per_word = [self.count(word) for word in words]
        out, i = [], 0
        while i < len(words):
            if per_word[i] > inner_budget:
                out.extend(self._split_atom(words[i], measure, emit))
                i += 1
                continue
            j, total = i, 0
            while j < len(words) and total + per_word[j] <= inner_budget:
                total += per_word[j]
                j += 1
            # verify the ASSEMBLED (and wrapped) string, shrinking on sub-word merging
            while j > i + 1 and measure(" ".join(words[i:j])) > self.budget:
                j -= 1
            candidate = " ".join(words[i:j])
            if measure(candidate) > self.budget:
                out.extend(self._split_atom(words[i], measure, emit))
                j = i + 1
            else:
                out.append(emit(candidate))
            i = j
        return out

    def _split_atom(self, word, measure, emit):
        """A single whitespace word that cannot fit alone: split it by characters.

        Binary search for the longest prefix that still fits, so the split is deterministic
        and minimal. GPQA's LaTeX formulas hit this (one 152-token "word").
        """
        pieces, start, n = [], 0, len(word)
        while start < n:
            lo, hi, best = 1, n - start, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if measure(word[start:start + mid]) <= self.budget:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                raise BudgetExhausted(
                    "a single character does not fit the %d-token budget; refusing to emit a "
                    "string the encoder would truncate" % self.budget)
            pieces.append(emit(word[start:start + best]))
            start += best
        return pieces


def build_chunker(encoder):
    """Chunker bound to THIS encoder's tokenizer and window."""
    return Chunker(encoder.count_tokens, encoder.token_budget())


def turn_chunks(argument, chunker):
    """One debater turn -> list of sentences, each a list of ready-to-encode chunk strings."""
    out = []
    for sentence in split_sentences(strip_quote_markup(argument)):
        chunks = chunker.chunk(sentence)
        if chunks:
            out.append(chunks)
    return out


_V_QUOTE_RE = re.compile(r"<v_quote>(.*?)</v_quote>", re.IGNORECASE | re.DOTALL)


def verified_quote_words(argument):
    """Words inside <v_quote>...</v_quote> ONLY.

    Plain <quote> and <u_quote> content is NOT verified evidence and is not counted here.
    This is why every argument is re-run through the production quote verifier first: the
    canonical merged datasets store plain <quote> tags, so counting pre-existing tag state
    would make this coordinate falsely constant at zero.
    """
    return sum(len(inner.split()) for inner in _V_QUOTE_RE.findall(argument))


def _build_uncertainty_regex(lexicon):
    # Longest alternative first so "it seems" beats "seems" at the same position and
    # "maybe" beats "may"; \b keeps matches on whole words / whole declared phrases.
    alts = sorted(lexicon, key=lambda t: (-len(t), t))
    body = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in alts)
    return re.compile(r"\b(?:%s)\b" % body, re.IGNORECASE)


_UNCERTAINTY_RE = _build_uncertainty_regex(UNCERTAINTY_LEXICON)


def uncertainty_matches(text):
    """(total_matches, {term: count}) over the quote-markup-stripped text.

    Case-insensitive, whole word / whole declared phrase, non-overlapping.
    """
    flat = normalise_whitespace(strip_quote_markup(text))
    counts = {}
    total = 0
    for match in _UNCERTAINTY_RE.finditer(flat):
        term = normalise_whitespace(match.group(0)).lower()
        counts[term] = counts.get(term, 0) + 1
        total += 1
    return total, counts


def _token_sublist_index(needle, haystack):
    """First index where the token list `needle` occurs contiguously in `haystack`, else -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    span = len(needle)
    for start in range(len(haystack) - span + 1):
        if haystack[start] == first and haystack[start:start + span] == needle:
            return start
    return -1


def hidden_answer_overlap(transcript, qh):
    """Exact and normalized hidden-answer overlap: a LEAKAGE DIAGNOSTIC, never a feature.

    Recorded so that `semantic_target_alignment` (an embedding cosine gap) can never be
    reported as though it were literal entailment of the hidden answer.

    Matching is TOKEN-BOUNDARY, not raw substring. GPQA hidden answers are frequently short
    ("4", "C6H6"), and a substring test reports those as leaked inside any unrelated number
    or formula that happens to contain them -- "4" matches "1234", "n=14", "0.45". Both the
    exact and the normalized test therefore require whole-token alignment.
    """
    plain = strip_quote_markup(at_common.transcript_text(transcript))
    plain_tokens = normalize_text(plain).split()
    out = {}
    for label in ("H_false", "H_true"):
        answer = (qh.get(label) or "").strip()
        answer_tokens = normalize_text(answer).split()
        exact = bool(answer and re.search(
            r"(?<!\w)%s(?!\w)" % re.escape(answer), plain))
        out["%s_exact_in_transcript" % label.lower()] = exact
        out["%s_normalized_in_transcript" % label.lower()] = bool(
            answer_tokens and _token_sublist_index(answer_tokens, plain_tokens) >= 0)
    return out


def residue_markers_in(transcript):
    text = at_common.transcript_text(transcript)
    return [marker for marker in RESIDUE_MARKERS if marker in text]


# ===========================================================================
# Encoders
#
# Two implementations behind one tiny contract: `encode(list[str]) -> (n, d) float
# array`, plus `provenance()` and `release()`. Tests and --dry-run inject the
# deterministic fake, which STAMPS ITSELF SYNTHETIC rather than impersonating MiniLM.
# ===========================================================================
def l2_normalize(vec):
    """Unit-normalize; returns None when the norm is not usable (empty / zero / non-finite)."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return arr / norm


def pool_normalized(vectors):
    """Mean of already-normalized vectors, re-normalized. None if it degenerates."""
    usable = [v for v in vectors if v is not None]
    if not usable:
        return None
    return l2_normalize(np.mean(np.vstack(usable), axis=0))


def pool_all_required(vectors):
    """Like pool_normalized, but a SINGLE undefined member makes the pool undefined.

    Used wherever the contract is "equal weight over exactly these members" (the two turns
    of a round, the 2R turns of a transcript): silently averaging the survivors would
    change the weighting without saying so, so the row is failed instead.
    """
    vectors = list(vectors)
    if not vectors or any(v is None for v in vectors):
        return None
    return pool_normalized(vectors)


class DeterministicFakeEncoder:
    """Synthetic encoder: SHA-256(text) -> seed -> fixed Gaussian vector.

    Deterministic across processes and machines, needs no download, no GPU and no network.
    It is stamped `synthetic: true` in every artifact it touches so its numbers can never
    be mistaken for the pinned production encoder's.
    """

    synthetic = True

    def __init__(self, dim=FAKE_ENCODER_DIM, name=FAKE_ENCODER_NAME,
                 seed_namespace="mi-post-fake-encoder", budget=FAKE_ENCODER_TOKEN_BUDGET):
        self.dim = int(dim)
        self.name = name
        self.seed_namespace = seed_namespace
        self.budget = int(budget)

    def token_budget(self):
        return self.budget

    def count_tokens(self, text):
        """Synthetic whitespace tokenizer, so the chunking path runs identically on CPU."""
        return len(text.split())

    def encode(self, texts):
        out = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(stable_seed(self.seed_namespace, self.dim, text))
            out[i] = rng.standard_normal(self.dim)
        return out

    def provenance(self):
        return {
            "synthetic": True,
            "model": self.name,
            "revision": None,
            "device": "cpu",
            "dim": self.dim,
            "batch_size": None,
            "implementation_version": EMBEDDING_IMPL_VERSION,
            "tokenizer": "synthetic-whitespace-words",
            "max_seq_length": self.budget,
            "special_tokens_reserved": 0,
            "token_budget": self.budget,
            "note": "deterministic synthetic embeddings; NOT the pinned production encoder",
        }

    def release(self):
        return False


class SentenceTransformerEncoder:
    """The pinned production encoder. sentence-transformers is imported LAZILY so the
    dry-run / test path never needs it installed."""

    synthetic = False

    def __init__(self, model=DEFAULT_EMBEDDING_MODEL, revision=DEFAULT_EMBEDDING_REVISION,
                 device=None, batch_size=DEFAULT_EMBEDDING_BATCH_SIZE):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.name = model
        self.revision = revision
        self.batch_size = int(batch_size)
        kwargs = {}
        if revision:
            kwargs["revision"] = revision
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model, **kwargs)
        self.model.eval()
        try:
            self.device = str(self.model.device)
        except AttributeError:
            self.device = device or "unknown"
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension; accept either so provenance never silently goes null.
        self.dim = None
        for accessor in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(self.model, accessor, None)
            if callable(getter):
                try:
                    self.dim = int(getter())
                    break
                except (TypeError, ValueError):
                    continue
        try:
            self.max_seq_length = int(self.model.max_seq_length)
        except (AttributeError, TypeError):
            self.max_seq_length = None
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None or self.max_seq_length is None:
            raise MiPostError(
                "encoder %r exposes no tokenizer/max_seq_length, so chunking could not be "
                "bound to its real window and long text would be silently truncated" % model)
        try:
            self.special_tokens_reserved = int(
                self.tokenizer.num_special_tokens_to_add(pair=False))
        except (AttributeError, TypeError):
            # Unknown scaffold size: reserve the BERT-family default rather than assume none.
            self.special_tokens_reserved = 2
        self.budget = self.max_seq_length - self.special_tokens_reserved
        if self.budget < 1:
            raise MiPostError(
                "encoder %r has no usable token budget (max_seq_length=%r, specials=%r)"
                % (model, self.max_seq_length, self.special_tokens_reserved))
        try:
            import sentence_transformers  # noqa: PLC0415
            self.library_version = sentence_transformers.__version__
        except Exception:                                    # noqa: BLE001
            self.library_version = None

    def token_budget(self):
        """Usable CONTENT tokens: the model window minus the specials the tokenizer adds."""
        return self.budget

    def count_tokens(self, text):
        # MEASUREMENT ONLY -- this call never produces model input. It is deliberately asked
        # to measure over-budget strings (that is how the chunker learns to split them and
        # how the audit records what the full text would have been), so transformers' "token
        # indices sequence length is longer than..." warning is noise here. `verbose=False`
        # gates only that logger.warning: ids, counts and truncation are untouched
        # (truncation stays False, so this is always the TRUE full length).
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False,
                                  verbose=False)["input_ids"])

    def encode(self, texts):
        vectors = self.model.encode(list(texts), batch_size=self.batch_size,
                                    convert_to_numpy=True, normalize_embeddings=False,
                                    show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float64)

    def provenance(self):
        return {
            "synthetic": False,
            "model": self.name,
            "revision": self.revision,
            "device": self.device,
            "dim": self.dim,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "special_tokens_reserved": self.special_tokens_reserved,
            "token_budget": self.budget,
            "tokenizer": type(self.tokenizer).__name__,
            "sentence_transformers_version": self.library_version,
            "implementation_version": EMBEDDING_IMPL_VERSION,
        }

    def release(self):
        self.model = None
        self.tokenizer = None
        import gc  # noqa: PLC0415
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                                # noqa: BLE001
                pass
        return True


def build_encoder(args):
    """Encoder factory seam. --dry-run always gets the explicitly synthetic encoder."""
    if args.dry_run:
        return DeterministicFakeEncoder()
    return SentenceTransformerEncoder(model=args.embedding_model,
                                      revision=args.embedding_revision,
                                      device=args.embedding_device,
                                      batch_size=args.embedding_batch_size)


class TextEmbedder:
    """Collect every text once, encode in ONE batched pass, then serve unit vectors.

    Encoding is the expensive step and many chunk strings repeat; deduplicating by exact
    string keeps the pass deterministic and cheap. Insertion order is the encode order.
    """

    def __init__(self, encoder, chunker=None):
        self.encoder = encoder
        self.chunker = chunker if chunker is not None else build_chunker(encoder)
        self.budget = self.chunker.budget
        self._order = []
        self._index = {}
        self._vectors = {}
        self.max_tokens_encoded = 0

    def add(self, text):
        if text not in self._index:
            self._index[text] = len(self._order)
            self._order.append(text)

    def add_many(self, texts):
        for text in texts:
            self.add(text)

    def tokens(self, text):
        """Memoised token count for `text` under the encoder's own tokenizer."""
        return self.chunker.count(text)

    def _assert_within_budget(self, pending):
        """THE gate: nothing over budget may reach encode(), where it would be truncated."""
        over = [(t, self.chunker.count(t)) for t in pending
                if self.chunker.count(t) > self.budget]
        if over:
            worst = max(over, key=lambda pair: pair[1])
            raise MiPostError(
                "%d staged text(s) exceed the encoder's %d-token budget (worst %d tokens: "
                "%r...); the encoder would truncate them silently, so the run is refused"
                % (len(over), self.budget, worst[1], worst[0][:80]))
        if pending:
            self.max_tokens_encoded = max(
                self.max_tokens_encoded, max(self.chunker.count(t) for t in pending))

    def run(self):
        if not self._order:
            return 0
        pending = [t for t in self._order if t not in self._vectors]
        if not pending:
            return 0
        self._assert_within_budget(pending)
        matrix = self.encoder.encode(pending)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(pending):
            raise MiPostError(
                "encoder returned shape %r for %d texts; expected (n, d)"
                % (getattr(matrix, "shape", None), len(pending)))
        for text, row in zip(pending, matrix):
            self._vectors[text] = l2_normalize(row)
        return len(pending)

    def vector(self, text):
        if text not in self._vectors:
            raise MiPostError("text was never staged for encoding: %r" % text[:80])
        return self._vectors[text]

    def n_texts(self):
        return len(self._order)


# ===========================================================================
# Dataset loading, identity and eligibility
# ===========================================================================
def _stable_pair_id(dataset, item):
    """The repo's order-independent pair identity (source: verifier-posterior-eval.py:123)."""
    dataset_slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")
    identity = json.dumps(
        [dataset, item["story_title"], item["Q_Y"]["question"], item["Q_H"]["question"]],
        ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return "%s_%s" % (dataset_slug, digest)


ANSWER_FIELDS = {"Q_Y": ("Y_true", "Y_false"), "Q_H": ("H_true", "H_false")}


def load_dataset(path, dataset_name):
    """Load the canonical dataset list and attach pair_id / dataset_index.

    Validates only the shape this file relies on: both question holders present with
    non-empty question + answer strings, and pair_id unique.
    """
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise MiPostError("%r must contain a top-level JSON array" % path)
    seen = set()
    for idx, item in enumerate(items):
        where = "item %d (%r)" % (idx, item.get("story_title", "?"))
        if not isinstance(item.get("story_title"), str) or not item["story_title"].strip():
            raise MiPostError("%s: empty/missing story_title" % where)
        for q_key in ("Q_Y", "Q_H"):
            holder = item.get(q_key)
            if not isinstance(holder, dict):
                raise MiPostError("%s: missing %s holder" % (where, q_key))
            for field in ("question",) + ANSWER_FIELDS[q_key]:
                if not isinstance(holder.get(field), str) or not holder[field].strip():
                    raise MiPostError("%s: empty/missing %s.%s" % (where, q_key, field))
        pair_id = item.get("pair_id") or _stable_pair_id(dataset_name, item)
        if pair_id in seen:
            raise MiPostError("%s: duplicate pair_id %r" % (where, pair_id))
        seen.add(pair_id)
        item["pair_id"] = pair_id
        item["dataset_index"] = idx
    return items


def load_story_map(path):
    with open(path, encoding="utf-8") as handle:
        stories = json.load(handle)
    if not isinstance(stories, dict):
        raise MiPostError("%r: story map must be a JSON object {title: story}" % path)
    return stories


def structural_transcript_check(transcript):
    """SHAPE only, no story access. None when usable, else the skip reason.

    Mirrors verifier-base-transcript-eval.py:structural_transcript_check exactly, including
    SKIP_EMPTY_ROUND for a zero-length debater argument (the gate that removes GPQA item 9's
    adversarial transcript and takes GPQA's common support from 83 to 82).
    """
    if not isinstance(transcript, dict):
        return SKIP_TRANSCRIPT_NULL
    stance = {name: transcript.get(name) for name in at_common.DEBATER_NAMES}
    if set(stance.values()) != set(at_common.STANCE_LABELS):
        return SKIP_STANCE_MALFORMED
    rounds = transcript.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return SKIP_ROUNDS_MISSING
    for rnd in rounds:
        if not isinstance(rnd, dict):
            return SKIP_EMPTY_ROUND
        for name in at_common.DEBATER_NAMES:
            argument = rnd.get(name)
            if not isinstance(argument, str) or not argument.strip():
                return SKIP_EMPTY_ROUND
    return None


def reverify_transcript(transcript, story):
    """Re-run the production quote verifier over EVERY stored debater argument.

    verify_quotes collapses any pre-existing <v_quote>/<u_quote> back to <quote> before
    re-tagging, so this is idempotent and never trusts stored tag state. The result is what
    both the feature extractors and render_verified_transcript() consume.
    """
    story_normalised = normalize_text(story)
    out = {name: transcript[name] for name in at_common.DEBATER_NAMES}
    out["rounds"] = [
        {name: verify_quotes(rnd[name], story_normalised) for name in at_common.DEBATER_NAMES}
        for rnd in transcript["rounds"]
    ]
    return out


def collect_condition(items, stories, condition):
    """-> (rows, skipped) for one condition, in dataset order."""
    field = TRANSCRIPT_FIELDS[condition]
    rows, skipped = [], []
    for item in items:
        transcript = (item.get("Q_Y") or {}).get(field)
        reason = structural_transcript_check(transcript)
        story = stories.get(item["story_title"])
        if reason is None and story is None:
            reason = SKIP_STORY_MISSING
        if reason is not None:
            skipped.append({"pair_id": item["pair_id"],
                            "dataset_index": item["dataset_index"],
                            "condition": condition, "reason": reason})
            continue
        reverified = reverify_transcript(transcript, story)
        rows.append({
            "pair_id": item["pair_id"],
            "dataset_index": item["dataset_index"],
            "story_title": item["story_title"],
            "condition": condition,
            "item": item,
            "story": story,
            "raw_transcript": transcript,
            "transcript": reverified,
            "raw_transcript_content_sha256": content_hash(transcript),
            "transcript_content_sha256": content_hash(reverified),
            "residue_markers": residue_markers_in(transcript),
        })
    return rows, skipped


def stance_parity_warnings(rows_by_condition):
    """Both-present pairs whose two conditions disagree on which debater defends Y_true.

    Recorded, never fatal: every feature here uses SEMANTIC stance roles, so a disagreement
    changes nothing numerically -- but it would mean the two arms were generated against
    different stance draws, which a reader must be told about.
    """
    by_pair = {}
    for condition, rows in rows_by_condition.items():
        for row in rows:
            by_pair.setdefault(row["pair_id"], {})[condition] = {
                name: row["transcript"][name] for name in at_common.DEBATER_NAMES}
    out = []
    for pair_id, stances in sorted(by_pair.items()):
        if len(stances) < len(CONDITIONS):
            continue
        values = list(stances.values())
        if any(v != values[0] for v in values[1:]):
            out.append({"pair_id": pair_id, "stance_by_condition": stances})
    return out


# ===========================================================================
# Features
# ===========================================================================
def target_anchor_wrapper(answer):
    """Wrap a question fragment into a COMPLETE target-anchor string.

    Every chunk of a long Q_H therefore carries the whole proposed answer. That repetition is
    deliberate: coordinate 1 is a contrast between the H_false and H_true anchors, so the
    answer has to be present in each embedded piece or the contrast would survive only in the
    first chunk and be diluted away by the rest. It does mean a long question's pooled anchor
    is answer-weighted; that is recorded, not hidden.
    """
    fixed = normalise_whitespace(answer)
    return lambda fragment: HYPOTHESIS_ANCHOR_TEMPLATE.format(question=fragment, answer=fixed)


def anchor_specs(item):
    """(text_to_chunk, wrapper) per anchor. The story is NEVER embedded."""
    qh, qy = item["Q_H"], item["Q_Y"]
    return {
        "a_hfalse": (qh["question"], target_anchor_wrapper(qh["H_false"])),
        "a_htrue": (qh["question"], target_anchor_wrapper(qh["H_true"])),
        "e_qh": (qh["question"], None),
        "e_qy": (qy["question"], None),
    }


def anchor_texts(item):
    """The four FULL anchor strings, unchunked. Provenance/hashing only -- never encoded
    directly, because a long one would be silently truncated by the encoder."""
    qh, qy = item["Q_H"], item["Q_Y"]
    return {
        "a_hfalse": HYPOTHESIS_ANCHOR_TEMPLATE.format(question=qh["question"],
                                                      answer=qh["H_false"]),
        "a_htrue": HYPOTHESIS_ANCHOR_TEMPLATE.format(question=qh["question"],
                                                     answer=qh["H_true"]),
        "e_qh": qh["question"],
        "e_qy": qy["question"],
    }


def stage_row(embedder, row, chunker):
    """Decompose one row into budget-checked chunks and stage every string for encoding.

    Transcript turns are sentence-split then chunked; anchors are chunked under their own
    wrapper. Anything that cannot be chunked at all is recorded per anchor and leaves that
    anchor undefined, which makes the dependent coordinates non-finite and drops the row
    through the ordinary eligibility ledger instead of embedding something truncated.
    """
    transcript = row["transcript"]
    per_round = []
    for rnd in transcript["rounds"]:
        turns = {}
        for name in at_common.DEBATER_NAMES:
            turns[name] = turn_chunks(rnd[name], chunker)
            for sentence in turns[name]:
                embedder.add_many(sentence)
        per_round.append(turns)
    row["chunks_by_round"] = per_round
    row["anchor_texts"] = anchor_texts(row["item"])
    row["anchor_chunks"] = {}
    row["anchor_chunk_errors"] = {}
    for key, (text, wrap) in anchor_specs(row["item"]).items():
        try:
            chunks = chunker.chunk(text, wrap=wrap)
        except BudgetExhausted as exc:
            row["anchor_chunk_errors"][key] = str(exc)
            row["anchor_chunks"][key] = []
            continue
        row["anchor_chunks"][key] = chunks
        embedder.add_many(chunks)
    return row


def _turn_vector(embedder, sentences):
    """chunk vectors -> sentence vectors -> turn vector, L2-normalizing after each level."""
    sentence_vectors = [pool_normalized([embedder.vector(chunk) for chunk in sentence])
                        for sentence in sentences]
    return pool_normalized(sentence_vectors)


def _cos(u, v):
    if u is None or v is None:
        return float("nan")
    return float(np.dot(u, v))


def compute_features(embedder, row):
    """Return the fixed six-feature vector and its reconstructing submeasurements."""
    transcript = row["transcript"]
    ytrue = ytrue_defender(transcript)
    yfalse = yfalse_defender(transcript)
    rounds = transcript["rounds"]

    # --- embedding pooling -------------------------------------------------
    turn_vectors = []
    round_vectors = []
    n_sentences = {ytrue: 0, yfalse: 0}
    n_chunks = {ytrue: 0, yfalse: 0}
    for turns in row["chunks_by_round"]:
        by_name = {}
        for name in at_common.DEBATER_NAMES:
            by_name[name] = _turn_vector(embedder, turns[name])
            n_sentences[name] += len(turns[name])
            n_chunks[name] += sum(len(sentence) for sentence in turns[name])
        # equal turn weight: verbosity must not be counted twice, length is coordinate 2
        turn_vectors.extend(by_name[name] for name in at_common.DEBATER_NAMES)
        round_vectors.append(pool_all_required(by_name[name]
                                               for name in at_common.DEBATER_NAMES))
    z_transcript = pool_all_required(turn_vectors)

    # Anchors pool exactly like a sentence: normalize each chunk, mean, renormalize. A short
    # anchor is one chunk, so this is identical to embedding the whole string.
    anchors = {key: pool_normalized([embedder.vector(chunk) for chunk in chunks])
               for key, chunks in row["anchor_chunks"].items()}

    # --- coordinate 1: semantic target alignment ---------------------------
    cos_hfalse = _cos(z_transcript, anchors["a_hfalse"])
    cos_htrue = _cos(z_transcript, anchors["a_htrue"])
    semantic_target_alignment = cos_hfalse - cos_htrue

    # --- coordinate 2: dequantized length imbalance ------------------------
    counts = round_word_counts(transcript)
    words_ytrue = int(sum(c[0] for c in counts))
    words_yfalse = int(sum(c[1] for c in counts))
    len_imbalance = len_imbalance_dequant(transcript, row["transcript_content_sha256"])

    # --- coordinate 3: verified evidence density asymmetry -----------------
    verified = {ytrue: 0, yfalse: 0}
    for rnd in rounds:
        for name in at_common.DEBATER_NAMES:
            verified[name] += verified_quote_words(rnd[name])
    density_ytrue = (verified[ytrue] / words_ytrue) if words_ytrue > 0 else float("nan")
    density_yfalse = (verified[yfalse] / words_yfalse) if words_yfalse > 0 else float("nan")
    verified_evidence_density_asymmetry = density_ytrue - density_yfalse

    # --- coordinate 4: semantic escalation ---------------------------------
    per_round_alignment = [_cos(vec, anchors["a_hfalse"]) - _cos(vec, anchors["a_htrue"])
                           for vec in round_vectors]
    semantic_escalation = (ols_slope(per_round_alignment) if len(per_round_alignment) >= 2
                           else float("nan"))

    # --- coordinate 5: uncertainty suppression -----------------------------
    unc_total = {ytrue: 0, yfalse: 0}
    unc_terms = {ytrue: {}, yfalse: {}}
    for rnd in rounds:
        for name in at_common.DEBATER_NAMES:
            total, terms = uncertainty_matches(rnd[name])
            unc_total[name] += total
            for term, count in terms.items():
                unc_terms[name][term] = unc_terms[name].get(term, 0) + count
    rate_ytrue = (unc_total[ytrue] / words_ytrue) if words_ytrue > 0 else float("nan")
    rate_yfalse = (unc_total[yfalse] / words_yfalse) if words_yfalse > 0 else float("nan")
    uncertainty_suppression = rate_yfalse - rate_ytrue

    # --- coordinate 6: hidden-task relevance -------------------------------
    cos_qh = _cos(z_transcript, anchors["e_qh"])
    cos_qy = _cos(z_transcript, anchors["e_qy"])
    hidden_task_relevance = cos_qh - cos_qy

    features = {
        "semantic_target_alignment": float(semantic_target_alignment),
        "len_imbalance_ytrue": float(len_imbalance),
        "verified_evidence_density_asymmetry": float(verified_evidence_density_asymmetry),
        "semantic_escalation": float(semantic_escalation),
        "uncertainty_suppression": float(uncertainty_suppression),
        "hidden_task_relevance": float(hidden_task_relevance),
    }
    submeasurements = {
        "ytrue_defender": ytrue,
        "yfalse_defender": yfalse,
        "n_rounds": len(rounds),
        "n_turns": 2 * len(rounds),
        "round_word_counts_ytrue_yfalse": [[int(a), int(b)] for a, b in counts],
        "words_ytrue": words_ytrue,
        "words_yfalse": words_yfalse,
        "total_words": words_ytrue + words_yfalse,
        "len_imbalance_ytrue_raw": len_imbalance_raw(transcript),
        "verified_quote_words_ytrue": int(verified[ytrue]),
        "verified_quote_words_yfalse": int(verified[yfalse]),
        "verified_density_ytrue": density_ytrue,
        "verified_density_yfalse": density_yfalse,
        "uncertainty_matches_ytrue": int(unc_total[ytrue]),
        "uncertainty_matches_yfalse": int(unc_total[yfalse]),
        "uncertainty_terms_ytrue": dict(sorted(unc_terms[ytrue].items())),
        "uncertainty_terms_yfalse": dict(sorted(unc_terms[yfalse].items())),
        "uncertainty_rate_ytrue": rate_ytrue,
        "uncertainty_rate_yfalse": rate_yfalse,
        "cos_transcript_hfalse": cos_hfalse,
        "cos_transcript_htrue": cos_htrue,
        "cos_transcript_qh": cos_qh,
        "cos_transcript_qy": cos_qy,
        "per_round_target_alignment": [float(v) for v in per_round_alignment],
        "n_sentences_ytrue": int(n_sentences[ytrue]),
        "n_sentences_yfalse": int(n_sentences[yfalse]),
        "n_chunks_ytrue": int(n_chunks[ytrue]),
        "n_chunks_yfalse": int(n_chunks[yfalse]),
        "n_sentences_chunked": int(sum(
            1 for turns in row["chunks_by_round"] for sentences in turns.values()
            for sentence in sentences if len(sentence) > 1)),
        "anchor_sha256": {key: sha256_text(text)
                          for key, text in row["anchor_texts"].items()},
        "anchor_word_counts": {key: len(text.split())
                               for key, text in row["anchor_texts"].items()},
        # Token/chunk audit: proves no anchor was silently truncated by the encoder.
        "anchor_chunk_counts": {key: len(chunks)
                                for key, chunks in row["anchor_chunks"].items()},
        "anchor_chunk_tokens": {key: [embedder.tokens(chunk) for chunk in chunks]
                                for key, chunks in row["anchor_chunks"].items()},
        "anchor_full_text_tokens": {key: embedder.tokens(text)
                                    for key, text in row["anchor_texts"].items()},
        "anchor_chunked": sorted(key for key, chunks in row["anchor_chunks"].items()
                                 if len(chunks) > 1),
        "anchor_chunk_errors": dict(row["anchor_chunk_errors"]),
        "transcript_chunk_tokens_max": max(
            [embedder.tokens(chunk) for turns in row["chunks_by_round"]
             for sentences in turns.values() for sentence in sentences for chunk in sentence]
            or [0]),
        "encoder_token_budget": embedder.budget,
        "transcript_text_sha256": sha256_text(at_common.transcript_text(transcript)),
        "raw_transcript_content_sha256": row["raw_transcript_content_sha256"],
        "transcript_content_sha256": row["transcript_content_sha256"],
        "hidden_answer_overlap": hidden_answer_overlap(transcript, row["item"]["Q_H"]),
        "n_turn_vectors_defined": int(sum(v is not None for v in turn_vectors)),
        "z_transcript_defined": z_transcript is not None,
        "round_vectors_defined": [vec is not None for vec in round_vectors],
    }
    row["features"] = features
    row["feature_submeasurements"] = submeasurements
    row["features_finite"] = all(np.isfinite(features[name]) for name in FEATURE_NAMES)
    return row


def feature_vector(row):
    return [float(row["features"][name]) for name in FEATURE_NAMES]


# ===========================================================================
# Checkpoints: fingerprints, metadata echo, family warnings
# ===========================================================================
def read_training_metadata(checkpoint_dir):
    path = os.path.join(checkpoint_dir, TRAINING_METADATA_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


METADATA_ECHO_KEYS = ("model_name", "dataset", "mode", "supervision", "qh_aux", "qh_lambda",
                      "qy_loss", "forced_choice_readout_input_suffix", "dataset_path",
                      "stories_path", "transcript_field", "base_dir", "output_dir",
                      "train_all", "train_items", "eval_items", "train_rows", "eval_rows",
                      "split_seed", "git_commit", "created_at")


def metadata_echo(metadata):
    """Verbatim echo of the scalar training-metadata fields that identify the arm."""
    if not isinstance(metadata, dict):
        return None
    return {key: metadata.get(key) for key in METADATA_ECHO_KEYS if key in metadata}


def checkpoint_family_warnings(echoes, dataset, mode, n_items, n_common):
    """Audit the fixed-base versus requested-mode checkpoint contract.

    None of these is fatal: the run's job is to compute one MI and say exactly what
    was on disk. Silence would be the failure mode.
    """
    warnings = []
    pre, post = echoes.get("pre"), echoes.get("post")
    if pre is None:
        warnings.append(
            "pre checkpoint has no readable %s; the path/fingerprint identifies V_base but "
            "its exact lineage cannot be verified from checkpoint metadata"
            % TRAINING_METADATA_FILE)
    if post is None:
        warnings.append(
            "post checkpoint has no readable %s; its requested %r training arm and lineage "
            "cannot be verified from the checkpoint itself"
            % (TRAINING_METADATA_FILE, mode))
    if pre and pre.get("mode") not in (None, "base", "pretrain", "pretrained"):
        warnings.append(
            "pre checkpoint metadata reports mode=%r, but this run requires the fixed V_base "
            "checkpoint" % pre.get("mode"))
    if post and post.get("mode") not in (None, mode):
        warnings.append(
            "post checkpoint metadata reports mode=%r, but --mode requests %r"
            % (post.get("mode"), mode))
    if post and post.get("qh_aux"):
        warnings.append(
            "post checkpoint metadata says qh_aux=%r (lambda=%r) with %r supervision: it is "
            "a Q_H-SUPERVISED arm, so a higher MI measures trained capacity, not emergent "
            "steering" % (post.get("qh_aux"), post.get("qh_lambda"), post.get("supervision")))
    for role in CHECKPOINT_ROLES:
        echo = echoes.get(role)
        if not echo:
            continue
        if echo.get("dataset") not in (None, dataset):
            warnings.append("%s checkpoint metadata names dataset %r, this run is %r"
                            % (role, echo.get("dataset"), dataset))
        train_items = echo.get("train_items")
        if isinstance(train_items, int) and train_items != n_items:
            warnings.append(
                "%s checkpoint metadata reports train_items=%d against the current dataset's "
                "%d items (common support %d); train/eval exposure is NOT reconstructed in v1"
                % (role, train_items, n_items, n_common))
    return warnings


def resolve_checkpoints(args, output_dir):
    """Fingerprint both checkpoints and refuse anything that cannot anchor a real run.

    A path is not an identity (training overwrites checkpoints in place), so real runs are
    bound to CONTENT fingerprints via adversarial_transcript.fingerprint
    (`resolve` -> `fingerprint` -> `validate_weights` + `require_production`). Identical
    fingerprints for pre and post would make r identically zero by construction, so that is
    a hard refusal too. --dry-run never touches a checkpoint: it gets two EXPLICIT,
    DISTINCT dry-run identities, because equal labels would fake r == 0.
    """
    paths = {"pre": args.pre_checkpoint, "post": args.post_checkpoint}
    labels = {"pre": "dry-run-base", "post": "dry-run-post-%s" % args.mode}
    # One fingerprint cache for the whole dataset run tree by default (see
    # resolve_fingerprint_cache_dir): the ~42 GB base checkpoint is shared by the honest and
    # adversarial arms, so rooting the cache in the per-mode output directory would re-hash
    # it once per arm before any GPU work started.
    cache_dir = (getattr(args, "fingerprint_cache_dir", None)
                 or fingerprint_mod.shared_cache_dir(output_dir))
    fingerprints = {}
    for role in CHECKPOINT_ROLES:
        fingerprints[role] = fingerprint_mod.resolve(
            paths[role], dry_run=args.dry_run, cache_dir=None if args.dry_run else cache_dir,
            label=labels[role])
    if fingerprint_mod.identity(fingerprints["pre"]) == fingerprint_mod.identity(
            fingerprints["post"]):
        raise MiPostError(
            "pre and post checkpoints have the SAME identity (%s); r = p_sem_post - p_sem_pre "
            "would be identically zero by construction. pre=%r post=%r"
            % (fingerprint_mod.identity(fingerprints["pre"]), paths["pre"], paths["post"]))
    return fingerprints


class ThreadSafeReadoutCache(ReadoutCache):
    """The existing append-only cache with one process-local lock.

    ``Reader`` performs a get, a GPU forward on miss, then a put.  The lock protects
    dictionary/stat counters and makes every JSONL append a complete fsync'd line.  A rare
    duplicate miss can still perform the same deterministic forward twice, but the second
    equal put is de-duplicated; a conflicting value for one content-bound key is refused.
    """

    def __init__(self, path=None):
        super().__init__(path)
        self._thread_lock = threading.RLock()
        self._key_locks = {}

    def get(self, key):
        with self._thread_lock:
            return super().get(key)

    def put(self, key, letter_logprobs):
        value = {letter: float(letter_logprobs[letter]) for letter in ("A", "B")}
        with self._thread_lock:
            existing = self._entries.get(key)
            if existing is not None:
                existing = {letter: float(existing[letter]) for letter in ("A", "B")}
                if existing != value:
                    raise MiPostError(
                        "concurrent readout cache conflict for key %s: %r != %r"
                        % (key, existing, value))
                return
            super().put(key, value)

    def contains(self, key):
        with self._thread_lock:
            return key in self._entries

    def stats(self):
        with self._thread_lock:
            return super().stats()

    @contextlib.contextmanager
    def lock_keys(self, keys):
        """Single-flight a row's prompt keys without serialising unrelated GPUs."""
        ordered = sorted(set(keys))
        with self._thread_lock:
            locks = [self._key_locks.setdefault(key, threading.RLock()) for key in ordered]
        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()


def _cuda_index(device):
    """Return a logical CUDA index, or ``None`` for a non-CUDA device."""
    if isinstance(device, bool):
        return None
    if isinstance(device, int):
        return int(device) if device >= 0 else None
    text = str(device).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
        return int(text.split(":", 1)[1])
    return None


def parse_devices(value):
    """Argparse converter for explicit logical CUDA devices (e.g. ``0,1,2,3``)."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("must name at least one logical CUDA device")
    devices = []
    for part in parts:
        index = _cuda_index(part)
        if index is None:
            raise argparse.ArgumentTypeError(
                "--devices accepts CUDA indices such as 0,1,2,3 or cuda:0,cuda:1")
        devices.append(index)
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices must not contain duplicate GPUs")
    return tuple(devices)


def validate_visible_cuda_devices(devices, torch_module=None):
    """Validate the one-replica-per-GPU contract before any verifier is loaded.

    A multi-worker invocation must own every CUDA device visible inside the process, using
    Slurm/CUDA's *logical* ``0..N-1`` indices.  This prevents an accidental partial pool or
    a physical GPU id after ``CUDA_VISIBLE_DEVICES`` remapping. A one-worker invocation
    also accepts one explicit visible CUDA index or a non-CUDA device.
    """
    indices = tuple(_cuda_index(device) for device in devices)
    if any(index is None for index in indices):
        if len(devices) != 1:
            raise MiPostError("a multi-worker readout pool requires CUDA devices only")
        return {"cuda": False, "visible_gpu_count": 0, "logical_devices": []}
    if len(set(indices)) != len(indices):
        raise MiPostError("readout worker devices must be unique, got %r" % (devices,))
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415
    if not torch_module.cuda.is_available():
        raise MiPostError(
            "CUDA is unavailable; the verifier readout needs %d visible GPU(s)"
            % len(indices))
    visible = int(torch_module.cuda.device_count())
    bad = [index for index in indices if index < 0 or index >= visible]
    if bad:
        raise MiPostError(
            "requested logical CUDA device(s) %r, but torch sees only %d device(s)"
            % (bad, visible))
    if len(indices) > 1:
        expected = tuple(range(len(indices)))
        if visible != len(indices) or set(indices) != set(expected):
            raise MiPostError(
                "multi-GPU readout requires exactly its worker GPUs to be visible as logical "
                "0..N-1; requested=%r visible=%d expected=%r. Set CUDA_VISIBLE_DEVICES or "
                "fix the Slurm allocation before loading checkpoints."
                % (indices, visible, expected))
    return {"cuda": True, "visible_gpu_count": visible,
            "logical_devices": list(indices)}


def _cuda_device_context(device, enabled=True):
    index = _cuda_index(device)
    if not enabled or index is None:
        return contextlib.nullcontext()
    import torch  # noqa: PLC0415
    return torch.cuda.device(index)


def _normalise_hf_destination(destination):
    """Normalise one HF placement destination to a CUDA index or return ``None``."""
    if isinstance(destination, bool):
        return None
    if isinstance(destination, int):
        return int(destination) if destination >= 0 else None
    destination_type = getattr(destination, "type", None)
    destination_index = getattr(destination, "index", None)
    if destination_type == "cuda" and destination_index is not None:
        return int(destination_index)
    rendered = str(destination).strip().lower()
    if rendered.isdigit():
        return int(rendered)
    if rendered.startswith("cuda:") and rendered.split(":", 1)[1].isdigit():
        return int(rendered.split(":", 1)[1])
    return None


def validate_single_gpu_placement(model, expected_device):
    """Fail closed unless the whole replica is on exactly one requested GPU.

    ``transformers`` does not guarantee that a whole-model, single-device
    ``device_map={"": i}`` load retains ``model.hf_device_map``.  In particular, the
    direct single-device path in Transformers 5.x can materialise every tensor on the
    requested GPU without dispatching the model (and therefore without attaching that
    metadata).  Prefer the map when it exists; otherwise audit every parameter and buffer
    rather than treating missing optional metadata as a placement failure.
    """
    expected = _cuda_index(expected_device)
    if expected is None:
        return {"method": "not_applicable", "used_cuda_devices": []}
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        if not isinstance(device_map, Mapping):
            raise MiPostError(
                "verifier exposed malformed hf_device_map %r; placement on cuda:%d "
                "cannot be audited" % (type(device_map).__name__, expected))
        used, invalid = set(), []
        for module_name, destination in device_map.items():
            index = _normalise_hf_destination(destination)
            if index is None:
                invalid.append("%s=%s" % (module_name, destination))
            else:
                used.add(index)
        if invalid:
            raise MiPostError(
                "verifier placement used a CPU/disk/non-CUDA destination: %s"
                % ", ".join(invalid[:4]))
        if used != {expected}:
            raise MiPostError(
                "verifier placement does not match cuda:%d (hf_device_map used %r)"
                % (expected, sorted(used)))
        return {
            "method": "hf_device_map",
            "used_cuda_devices": sorted(used),
            "parameters_scanned": None,
            "buffers_scanned": None,
        }

    tensor_getters = (
        ("parameter", getattr(model, "named_parameters", None)),
        ("buffer", getattr(model, "named_buffers", None)),
    )
    missing_getters = [kind for kind, getter in tensor_getters if not callable(getter)]
    if missing_getters:
        raise MiPostError(
            "verifier load exposed no hf_device_map and no auditable named_%s; "
            "placement on cuda:%d cannot be verified"
            % ("/named_".join(missing_getters), expected))

    used, invalid = set(), []
    counts = {"parameter": 0, "buffer": 0}
    try:
        for kind, getter in tensor_getters:
            for name, tensor in getter(recurse=True):
                counts[kind] += 1
                destination = getattr(tensor, "device", None)
                index = _normalise_hf_destination(destination)
                if index is None:
                    invalid.append("%s:%s=%s" % (kind, name, destination))
                else:
                    used.add(index)
                    if index != expected:
                        invalid.append("%s:%s=%s" % (kind, name, destination))
    except Exception as exc:
        raise MiPostError(
            "verifier load exposed no hf_device_map and tensor placement scan failed: "
            "%s: %s" % (type(exc).__name__, exc)) from exc

    n_tensors = counts["parameter"] + counts["buffer"]
    if n_tensors == 0:
        raise MiPostError(
            "verifier load exposed no hf_device_map and no parameters or buffers; "
            "placement on cuda:%d cannot be verified" % expected)
    if invalid:
        raise MiPostError(
            "verifier tensor placement used a CPU/meta/non-CUDA or foreign-GPU "
            "destination (expected cuda:%d): %s"
            % (expected, ", ".join(invalid[:4])))
    if used != {expected}:
        raise MiPostError(
            "verifier tensor placement does not match cuda:%d (tensor scan used %r)"
            % (expected, sorted(used)))
    return {
        "method": "tensor_device_scan",
        "used_cuda_devices": sorted(used),
        "parameters_scanned": counts["parameter"],
        "buffers_scanned": counts["buffer"],
    }


class LazyForcedChoiceVerifier:
    """A ForcedChoiceVerifier that is only CONSTRUCTED on the first cache miss.

    A warm-cache rerun must perform zero model forwards; loading 20B of weights just to
    discover every prompt is already cached would make "resume" meaningless. Duck-typed to
    what readout.Reader / release_scorer actually use: `choice_logprobs` and `release`.
    """

    def __init__(self, model_name, device=0):
        self.model_name = model_name
        self.device = device
        self.quantized_load = None
        self.scorer = None
        self.loads = 0
        self.placement_audit = None

    def _ensure(self):
        if self.scorer is None:
            # Keep the lock at the actual HF construction boundary.  The pool loads
            # replicas serially already, but this also protects direct/concurrent callers.
            with _HF_LOAD_LOCK:
                if self.scorer is None:
                    from adversarial_transcript.score_verifier import (  # noqa: PLC0415
                        ForcedChoiceVerifier,
                    )
                    with _cuda_device_context(self.device):
                        scorer = ForcedChoiceVerifier(self.model_name, device=self.device)
                    self.placement_audit = validate_single_gpu_placement(
                        scorer.model, self.device)
                    self.scorer = scorer
                    self.quantized_load = scorer.quantized_load
                    self.loads += 1
        return self.scorer

    def load(self):
        """Materialise the replica before any item-level threads begin forwarding."""
        self._ensure()
        return self

    def choice_logprobs(self, prompt):
        return self._ensure().choice_logprobs(prompt)

    def release(self, log=None):
        released = release_scorer(self.scorer, log=log) if self.scorer is not None else False
        self.scorer = None
        return released


def default_scorer_factory(checkpoint_path, device):
    return LazyForcedChoiceVerifier(checkpoint_path, device=device)


# ===========================================================================
# Readout: dual-order, no-CoT, forced choice on H_false
# ===========================================================================
def build_candidate(row):
    """The minimal candidate adapter the production prompt builders expect.

    `item_id := pair_id`: deterministic_answer_order keys on item_id, and BOTH orders are
    always scored and averaged, so the keying only fixes which order is LABELLED 'AB'. It
    cannot move p_sem.
    """
    item = row["item"]
    return {
        "item_id": row["pair_id"],
        "pair_id": row["pair_id"],
        "story_title": row["story_title"],
        "condition": row["condition"],
        "qy": {"question": item["Q_Y"]["question"],
               "Y_true": item["Q_Y"]["Y_true"], "Y_false": item["Q_Y"]["Y_false"]},
        "qh": {"question": item["Q_H"]["question"],
               "H_true": item["Q_H"]["H_true"], "H_false": item["Q_H"]["H_false"]},
        "transcript": row["transcript"],
    }


def prepare_prompts(row, option_seed):
    """Render ONCE per row and build both Q_H prompts; the story stops here.

    render_verified_transcript re-runs verify_quotes over the already-reverified strings,
    which is idempotent, so the scored transcript block is byte-identical to what the
    feature extractors read.
    """
    candidate = build_candidate(row)
    rendered = render_verified_transcript(candidate, row["story"])
    orders = orders_for(candidate, "Q_H", option_seed)
    prompts = {tag: qh_prompt(candidate, order, rendered) for tag, order in orders.items()}
    row["candidate"] = candidate
    row["rendered_sha256"] = sha256_text(rendered)
    row["rendered_chars"] = len(rendered)
    row["orders"] = {tag: dict(order) for tag, order in orders.items()}
    row["prompts"] = prompts
    row["prompt_sha256"] = {tag: prompt_hash(prompt) for tag, prompt in prompts.items()}
    return row


def _order_tag(order, orders_by_tag, pair_id):
    """Which tag ('AB'/'BA') this order dict is, by exact semantic content."""
    for tag, known in orders_by_tag.items():
        if dict(known) == dict(order):
            return tag
    raise MiPostError("unrecognised answer order %r for pair %r" % (order, pair_id))


def score_row(reader, row, role):
    """Score one item with one checkpoint (both orders stay on the same worker)."""
    candidate = row["candidate"]

    def prompt_fn(order, _row=row):
        return _row["prompts"][_order_tag(order, _row["orders"], _row["pair_id"])]

    scorer_id = reader.scorer_id()
    cache_keys = [
        ReadoutCache.key(scorer_id, reader.option_seed, reader.template_set, sha)
        for sha in row["prompt_sha256"].values()
    ]
    lock_keys = getattr(reader.cache, "lock_keys", None)
    lock_context = lock_keys(cache_keys) if callable(lock_keys) else contextlib.nullcontext()
    with lock_context:
        block = reader.block(candidate, "Q_H", prompt_fn, hypothesis=HYPOTHESIS)
    if block.get("hypothesis") != HYPOTHESIS:
        raise MiPostError(
            "readout returned hypothesis %r, expected %r: p_sem would be the posterior "
            "on the wrong hypothesis" % (block.get("hypothesis"), HYPOTHESIS))
    row.setdefault("readout", {})[role] = block
    return row


def score_rows(reader, rows, role):
    """Serial compatibility helper used by focused readout tests.

    Reader.block re-derives the two orders itself and calls back for each prompt; the
    callback serves the SAME precomputed prompt strings both checkpoints see, which is what
    makes the per-row prompt-SHA equality check below meaningful rather than tautological
    at the rendering level.
    """
    for row in rows:
        score_row(reader, row, role)
    return rows


def _cache_contains(cache, key):
    contains = getattr(cache, "contains", None)
    if callable(contains):
        return bool(contains(key))
    return key in getattr(cache, "_entries", {})


def checkpoint_cache_complete(cache, checkpoint_fingerprint, option_seed, rows):
    scorer_id = fingerprint_mod.identity(checkpoint_fingerprint)
    for row in rows:
        for sha in row["prompt_sha256"].values():
            key = ReadoutCache.key(scorer_id, option_seed, TEMPLATE_SET_ID, sha)
            if not _cache_contains(cache, key):
                return False
    return True


class ResidentReadoutWorker:
    """One exclusive scorer replica bound to one logical GPU."""

    def __init__(self, worker_id, device, reader, scorer, use_cuda_context=False,
                 validate_placement=False):
        self.worker_id = int(worker_id)
        self.device = device
        self.reader = reader
        self.scorer = scorer
        self.use_cuda_context = bool(use_cuda_context)
        self.validate_placement = bool(validate_placement)
        self.placement_audit = None
        self.rows_scored = 0
        self.released = False

    def load(self):
        if self.scorer is None:
            return
        with _cuda_device_context(self.device, self.use_cuda_context):
            loader = getattr(self.scorer, "load", None)
            if callable(loader):
                loader()
            if self.validate_placement:
                inner = getattr(self.scorer, "scorer", self.scorer)
                model = getattr(inner, "model", None)
                if model is None:
                    raise MiPostError(
                        "worker %d on %r loaded no auditable model object"
                        % (self.worker_id, self.device))
                self.placement_audit = validate_single_gpu_placement(model, self.device)

    def score(self, row, role):
        if self.released:
            raise MiPostError("readout worker %d was already released" % self.worker_id)
        with _cuda_device_context(self.device, self.use_cuda_context):
            score_row(self.reader, row, role)
        self.rows_scored += 1
        return row["pair_id"]

    def metrics(self):
        return {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "rows_scored": int(self.rows_scored),
            "readout_calls": int(self.reader.calls),
            "model_loads": int(getattr(self.scorer, "loads", 0) or 0),
            "placement_audit": self.placement_audit,
        }

    def release(self):
        if self.released:
            return False
        with _cuda_device_context(self.device, self.use_cuda_context):
            released = self.reader.release(log=None)
        self.released = True
        return released


def score_checkpoint_rows(rows, role, checkpoint_path, label, checkpoint_fingerprint,
                          option_seed, cache, devices, dry_run, scorer_factory,
                          use_cuda_context=False, validate_placement=False, log=print):
    """Score one checkpoint through a dynamic item queue and resident replicas.

    Models are constructed/materialised serially.  Only after every requested replica is
    resident does a ``ThreadPoolExecutor`` begin item-level work.  The queue leases each
    worker exclusively, so both answer-order forwards for one item use one model/GPU and no
    model object is ever called concurrently.  Input row order is never mutated.
    """
    devices = tuple(devices)
    if not devices:
        raise MiPostError("readout worker pool needs at least one device")
    workers = []
    try:
        # Construction is intentionally serial.  The default factory is lazy, preserving
        # zero model loads on a fully warm cache; an injected eager factory is also called
        # serially and under the target CUDA context.
        for worker_id, device in enumerate(devices):
            if dry_run:
                scorer = None
            else:
                with _HF_LOAD_LOCK:
                    with _cuda_device_context(device, use_cuda_context):
                        scorer = scorer_factory(checkpoint_path, device)
            reader = Reader(
                scorer=scorer, model_label=label, option_seed=option_seed,
                cache=cache, dry_run=dry_run,
                checkpoint_fingerprint=checkpoint_fingerprint)
            workers.append(ResidentReadoutWorker(
                worker_id, device, reader, scorer,
                use_cuda_context=use_cuda_context,
                validate_placement=validate_placement))

        cache_complete = checkpoint_cache_complete(
            cache, checkpoint_fingerprint, option_seed, rows)
        if not dry_run and not cache_complete:
            log("[mi-post] %s: loading %d verifier replica(s) serially on %s"
                % (role, len(workers), ",".join("cuda:%s" % _cuda_index(d)
                                                for d in devices)))
            for worker in workers:
                worker.load()

        log("[mi-post] %s: scoring %d item(s) with %d resident worker(s)%s"
            % (role, len(rows), len(workers),
               " (cache-only; no model load)" if cache_complete else ""))
        available = queue.Queue()
        for worker in workers:
            available.put(worker)
        stop = threading.Event()

        def dispatch(row):
            if stop.is_set():
                return None
            worker = available.get()
            try:
                if stop.is_set():
                    return None
                return worker.score(row, role)
            finally:
                available.put(worker)

        # A first failure cancels every not-yet-started sibling. Those cancellations then
        # surface here as CancelledError, which is NOT a diagnosis: reporting them next to
        # the real cause buried it under ~N lines of noise in the only forensic artifact a
        # cluster run leaves behind. Real failures and cancellations are therefore counted
        # separately, and only the real ones are quoted.
        errors = []
        n_cancelled = 0
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(workers), thread_name_prefix="mi-post-gpu") as executor:
            futures = {executor.submit(dispatch, row): row for row in rows}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    # A sibling we cancelled after someone else's failure. It never ran, so
                    # it has no cause of its own and nothing of its own was left half-written.
                    n_cancelled += 1
                except BaseException as exc:  # keep completed cache lines resume-safe
                    stop.set()
                    errors.append(
                        "%s/%s: %s: %s"
                        % (row.get("pair_id"), row.get("condition"),
                           type(exc).__name__, exc))
                    for sibling in futures:
                        if sibling is not future:
                            sibling.cancel()
        if errors:
            detail = "%s checkpoint worker pool failed (%d item failure(s)):\n  %s" % (
                role, len(errors), "\n  ".join(sorted(errors)))
            if n_cancelled:
                detail += ("\n  ... plus %d sibling item(s) cancelled after the first "
                           "failure (no cause of their own; not scored)" % n_cancelled)
            raise MiPostError(detail)
        if n_cancelled:
            # Cancellation without any recorded failure would mean rows silently went
            # unscored, which must never be reported as a completed phase.
            raise MiPostError(
                "%s checkpoint worker pool cancelled %d item(s) without recording a cause; "
                "refusing to report an incomplete phase" % (role, n_cancelled))

        worker_metrics = [worker.metrics() for worker in workers]
        base_provenance = dict(workers[0].reader.provenance())
        base_provenance["execution"] = {
            "version": READOUT_EXECUTION_VERSION,
            "scheduler": "ThreadPoolExecutor dynamic item queue",
            "item_granularity": "one selected transcript row; both orders on one worker",
            "requested_workers": len(workers),
            "logical_devices": [str(device) for device in devices],
            "cache_complete_before_phase": bool(cache_complete),
            "gpu_forwarding_required": bool(not dry_run and not cache_complete),
            "all_requested_replicas_resident_before_forwarding": (
                None if (dry_run or cache_complete)
                else bool(all(metric["model_loads"] > 0 for metric in worker_metrics))),
            "workers": worker_metrics,
        }
        return {
            "provenance": base_provenance,
            "readout_calls": int(sum(metric["readout_calls"] for metric in worker_metrics)),
            "model_loads": int(sum(metric["model_loads"] for metric in worker_metrics)),
        }
    finally:
        primary_failure_in_flight = sys.exc_info()[0] is not None
        release_errors = []
        for worker in workers:
            try:
                worker.release()
            except Exception as exc:  # noqa: BLE001 - release all siblings before refusing
                release_errors.append(
                    "worker %d on %r: %s: %s"
                    % (worker.worker_id, worker.device, type(exc).__name__, exc))
        if release_errors and not primary_failure_in_flight:
            raise MiPostError(
                "readout worker release failed:\n  %s"
                % "\n  ".join(release_errors))


def assemble_response(row):
    """r = p_sem^post - p_sem^pre, after the per-pair invariants. -> (ok, reason)."""
    readout = row.get("readout") or {}
    for role in CHECKPOINT_ROLES:
        block = readout.get(role)
        if not isinstance(block, dict):
            return False, "missing_%s_block" % role
        if set(block.get("orders") or {}) != {"AB", "BA"}:
            return False, "%s_missing_an_answer_order" % role
        for field in ("l_A", "l_B", "m", "p_sem"):
            value = block.get(field)
            if value is None or not np.isfinite(float(value)):
                return False, "%s_nonfinite_%s" % (role, field)
    pre, post = readout["pre"], readout["post"]
    if pre["prompt_sha256_by_order"] != post["prompt_sha256_by_order"]:
        return False, "prompt_sha_mismatch_pre_vs_post"
    if pre["prompt_sha256_by_order"] != row["prompt_sha256"]:
        return False, "prompt_sha_mismatch_vs_precomputed"
    r = float(post["p_sem"]) - float(pre["p_sem"])
    if not np.isfinite(r):
        return False, "nonfinite_response"
    row["response"] = {
        "r": r,
        "p_sem": {role: float(readout[role]["p_sem"]) for role in CHECKPOINT_ROLES},
        "l_sem": {role: float(readout[role]["m"]) for role in CHECKPOINT_ROLES},
        "l_pos": {role: float(readout[role]["l_pos"]) for role in CHECKPOINT_ROLES},
        "l_A": {role: float(readout[role]["l_A"]) for role in CHECKPOINT_ROLES},
        "l_B": {role: float(readout[role]["l_B"]) for role in CHECKPOINT_ROLES},
        # per-order log-odds of H_false, keyed by the order tag that was scored
        "logodds_by_order": {
            role: {tag: float(block["logodds"])
                   for tag, block in readout[role]["orders"].items()}
            for role in CHECKPOINT_ROLES},
        "max_letter_logprob": {role: (None if readout[role].get("max_letter_logprob") is None
                                      else float(readout[role]["max_letter_logprob"]))
                               for role in CHECKPOINT_ROLES},
        "letter_logprobs": {
            role: {tag: dict(block["letter_logprobs"])
                   for tag, block in readout[role]["orders"].items()}
            for role in CHECKPOINT_ROLES},
        "hypothesis_letter": {
            role: {tag: block["hypothesis_letter"]
                   for tag, block in readout[role]["orders"].items()}
            for role in CHECKPOINT_ROLES},
    }
    return True, None


# ===========================================================================
# Analysis
# ===========================================================================
def _duplicate_count(values):
    """Rows sharing an identical value with at least one other row (KSG continuity check)."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return int(sum(n for n in counts.values() if n > 1))


def condition_diagnostics(rows):
    """Recorded, never analysed and never gating."""
    feature_keys = [tuple(round(v, 12) for v in feature_vector(row)) for row in rows]
    r_values = [round(row["response"]["r"], 12) for row in rows]
    out = {
        "n_rows": len(rows),
        "n_duplicate_feature_rows": _duplicate_count(feature_keys),
        "n_duplicate_response_values": _duplicate_count(r_values),
        "n_residue_flagged": int(sum(1 for row in rows if row["residue_markers"])),
        "per_feature": {},
        "per_checkpoint": {},
    }
    for i, name in enumerate(FEATURE_NAMES):
        column = np.array([feature_vector(row)[i] for row in rows], dtype=float)
        out["per_feature"][name] = {
            "n_unique": int(len(set(np.round(column, 12).tolist()))),
            "min": float(column.min()) if column.size else None,
            "max": float(column.max()) if column.size else None,
            "mean": float(column.mean()) if column.size else None,
            "sd": float(column.std(ddof=1)) if column.size > 1 else 0.0,
        }
    for role in CHECKPOINT_ROLES:
        p_sem = np.array([row["response"]["p_sem"][role] for row in rows], dtype=float)
        l_sem = np.array([row["response"]["l_sem"][role] for row in rows], dtype=float)
        health = [row["response"]["max_letter_logprob"][role] for row in rows]
        health = [v for v in health if v is not None]
        out["per_checkpoint"][role] = {
            "n_saturated_p_sem": int(np.sum((p_sem <= SATURATION_EPS) |
                                            (p_sem >= 1.0 - SATURATION_EPS))),
            "saturation_eps": SATURATION_EPS,
            "p_sem_min": float(p_sem.min()) if p_sem.size else None,
            "p_sem_max": float(p_sem.max()) if p_sem.size else None,
            "abs_l_sem_max": float(np.abs(l_sem).max()) if l_sem.size else None,
            "max_letter_logprob_min": min(health) if health else None,
            "max_letter_logprob_max": max(health) if health else None,
            "n_suspect_max_letter_logprob": int(sum(1 for v in health
                                                    if v < SUSPECT_MAX_LETTER_LOGPROB)),
            "suspect_threshold": SUSPECT_MAX_LETTER_LOGPROB,
        }
    r_array = np.array([row["response"]["r"] for row in rows], dtype=float)
    out["response"] = {
        "min": float(r_array.min()) if r_array.size else None,
        "max": float(r_array.max()) if r_array.size else None,
        "mean": float(r_array.mean()) if r_array.size else None,
        "sd": float(r_array.std(ddof=1)) if r_array.size > 1 else 0.0,
        "n_unique": int(len(set(np.round(r_array, 12).tolist()))),
    }
    return out


def analyse(rows, k, mode):
    """Robust-scale and estimate the one mode-specific KSG quantity for this invocation."""
    if mode not in MODE_CHOICES:
        raise MiPostError("unknown mode %r (expected one of %r)" % (mode, MODE_CHOICES))
    condition = MODE_TO_CONDITION[mode]
    order = [row["pair_id"] for row in rows]
    if len(order) != len(set(order)):
        raise MiPostError("duplicate pair_id in %s analysis rows" % condition)
    wrong = sorted({row.get("condition") for row in rows
                    if row.get("condition") != condition})
    if wrong:
        raise MiPostError(
            "--mode %r requires condition %r rows, but analysis also received %r"
            % (mode, condition, wrong))
    n = len(order)
    F = np.asarray([feature_vector(row) for row in rows], dtype=float).reshape(
        (n, len(FEATURE_NAMES)))
    r = np.asarray([row["response"]["r"] for row in rows], dtype=float).reshape((n,))
    if F.shape != (n, len(FEATURE_NAMES)):
        raise MiPostError("F_%s has shape %r, expected (%d, %d)"
                          % (condition, F.shape, n, len(FEATURE_NAMES)))
    if r.shape != (n,):
        raise MiPostError("r_%s has shape %r, expected (%d,)"
                          % (condition, r.shape, n))

    analysis = {
        "estimator": ESTIMATOR_VERSION,
        "estimator_detail": ("KSG1 (Kraskov-Stogbauer-Grassberger, first estimator), "
                             "Chebyshev norm, strict '<' marginal counts, raw estimates "
                             "with negatives retained, units = nats"),
        "k": int(k),
        "n_pairs": n,
        "mode": mode,
        "condition": condition,
        "feature_names": list(FEATURE_NAMES),
        "pair_ids": list(order),
    }
    if n == 0:
        analysis.update({"mi": None, "mi_skipped_reason": "empty_population",
                         "skipped_reason": "empty_population", "scaler": None,
                         "kept_features": [],
                         "constant_feature_drops": list(FEATURE_NAMES),
                         "response_coordinate_kept": False,
                         "diagnostics": condition_diagnostics(rows)})
        return analysis

    stacked = np.column_stack([F, r])
    params = robust_scale_params(stacked)
    med, scale, keep = params
    kept_features = [name for name, flag in zip(FEATURE_NAMES, keep[:len(FEATURE_NAMES)])
                     if flag]
    constant_drops = [name for name, flag in zip(FEATURE_NAMES, keep[:len(FEATURE_NAMES)])
                      if not flag]
    response_kept = bool(keep[-1])
    analysis["scaler"] = {
        "policy": ("median / 1.4826*MAD; MAD=0 and var>0 -> std; both zero -> constant "
                   "coordinate dropped from the KSG input and recorded"),
        "fit_on": "selected %s mode rows (N x 7)" % mode,
        # Machine-readable form of the comparability caveat: the contract requires this
        # scaler to see only the selected mode's observations, so the two mode runs share a
        # population but NOT a coordinate metric.
        "jointly_fitted_across_modes": False,
        "n_fit_rows": int(stacked.shape[0]),
        "center": [float(v) for v in med],
        "scale": [float(v) for v in scale],
        "keep": [bool(v) for v in keep],
        "columns": list(FEATURE_NAMES) + ["r"],
    }
    analysis["kept_features"] = kept_features
    analysis["constant_feature_drops"] = constant_drops
    analysis["response_coordinate_kept"] = response_kept

    global_skip = None
    if not response_kept:
        global_skip = ("response r is exactly constant on the selected support; MI is "
                       "undefined for a constant coordinate")
    elif not kept_features:
        global_skip = ("every feature coordinate is exactly constant on the selected support; "
                       "at least one nonconstant coordinate is required")
    elif n <= int(k):
        global_skip = "N (%d) <= k (%d); KSG1 requires k < N" % (n, int(k))

    mi = None
    skipped = global_skip
    if global_skip is None:
        Z = apply_scale(stacked, params)
        try:
            mi = ksg_mi(Z[:, :-1], Z[:, -1], int(k))
        except ValueError as exc:
            skipped = "ksg refused the %s condition: %s" % (condition, exc)
    analysis["mi"] = mi
    analysis["mi_skipped_reason"] = skipped
    analysis["skipped_reason"] = skipped
    analysis["diagnostics"] = condition_diagnostics(rows)
    return analysis


# ===========================================================================
# Writers (atomic, schema-versioned, readout-string gated)
# ===========================================================================
def atomic_write_json(path, obj):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def assert_readout_family():
    """This file's readout string may never collide with an incompatible family."""
    for prefix in FORBIDDEN_READOUT_PREFIXES:
        if READOUT_ID.startswith(prefix):
            raise MiPostError(
                "READOUT_ID %r collides with the incompatible %r family" % (READOUT_ID, prefix))


def _is_present(path):
    """A file that exists with content. A zero-byte file is treated as absent (nothing was
    ever committed there); anything with bytes in it must identify itself."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError as exc:
        raise MiPostError("cannot stat %r (%s); refusing to guess whether it is ours"
                          % (path, exc)) from exc


def check_output_dir_readout(output_dir, dataset, mode):
    """Refuse to write next to artifacts from another readout, dataset, or mode.

    FAIL CLOSED. An existing non-empty artifact must parse as a JSON object carrying OUR
    readout id. Corrupt, unreadable, non-object, missing-the-field and foreign all raise:
    each of those is an artifact of unknown provenance, and overwriting one would destroy
    evidence while silently claiming its directory. A directory with no artifacts, or with
    zero-byte ones, is a normal fresh output directory and is left alone.
    """
    for name in ("observations.json", "results.json"):
        path = os.path.join(output_dir, name)
        if not _is_present(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, ValueError) as exc:
            raise MiPostError(
                "%r exists but is unreadable/corrupt (%s: %s). Refusing to overwrite an "
                "artifact whose provenance cannot be established -- move it aside or choose "
                "another --output-dir." % (path, type(exc).__name__, exc)) from exc
        if not isinstance(obj, dict) or "readout_id" not in obj:
            raise MiPostError(
                "%r exists but carries no readout_id, so it cannot be shown to belong to "
                "this readout family. Refusing to overwrite it." % path)
        if obj["readout_id"] != READOUT_ID:
            raise MiPostError(
                "%s in %r was written by readout %r, this run is %r. Readout families are "
                "not comparable and must never share an output directory."
                % (name, output_dir, obj["readout_id"], READOUT_ID))
        for key, expected in (("dataset", dataset), ("mode", mode)):
            if key not in obj:
                raise MiPostError(
                    "%r exists but carries no %s, so it cannot be shown to belong to this "
                    "single-mode estimand. Refusing to overwrite it." % (path, key))
            if obj[key] != expected:
                raise MiPostError(
                    "%s in %r was written with %s=%r, this run uses %r. Use a separate "
                    "--output-dir for every dataset/mode."
                    % (name, output_dir, key, obj[key], expected))


def cache_meta_path(cache_path):
    return "%s.meta.json" % cache_path


def check_cache_identity(cache_path, option_seed, dataset, mode):
    """Bind the readout cache to this readout id / template set / option seed / dataset / mode.

    The cache is keyed by (scorer identity, option seed, template set, prompt SHA), which
    is safe within a family but says nothing about WHICH family wrote it. This sidecar
    makes a cross-family reuse an error instead of a silent pool.

    DATASET AND MODE ARE PART OF THE IDENTITY (v2). Content-bound keys already make it
    impossible to SERVE one population's logits to another -- prompt SHAs differ, so a
    foreign entry can only miss. But a cache shared across datasets/modes silently
    accumulates foreign entries, and the `cache_stats` stamped into results then reports an
    entry count that has nothing to do with this run's population. That is misleading
    provenance on a number a paper will quote, so the sidecar refuses the sharing outright.
    The DEFAULT cache lives inside the dataset/mode-gated output directory; this gate is
    what closes the explicit `--cache` override.
    """
    if dataset not in DATASET_CHOICES or mode not in MODE_CHOICES:
        raise MiPostError(
            "cache identity needs a valid dataset/mode, got dataset=%r mode=%r"
            % (dataset, mode))
    meta = {
        "schema_version": CACHE_META_SCHEMA_VERSION,
        "readout_id": READOUT_ID,
        "readout_family": READOUT_FAMILY,
        "template_set": TEMPLATE_SET_ID,
        "template_set_sha256": TEMPLATE_SET_SHA256,
        "option_seed": int(option_seed),
        "dataset": dataset,
        "mode": mode,
    }
    path = cache_meta_path(cache_path)
    required = ("schema_version", "readout_id", "template_set_sha256", "option_seed",
                "dataset", "mode")
    if _is_present(path):
        try:
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, ValueError) as exc:
            raise MiPostError(
                "readout cache sidecar %r is unreadable/corrupt (%s: %s); the cache it "
                "describes cannot be trusted and will not be reused or overwritten."
                % (path, type(exc).__name__, exc)) from exc
        if not isinstance(existing, dict):
            raise MiPostError("readout cache sidecar %r is not a JSON object" % path)
        recorded_schema = existing.get("schema_version")
        if recorded_schema != CACHE_META_SCHEMA_VERSION:
            raise MiPostError(
                "readout cache sidecar %r declares schema_version=%r, this run writes %r. "
                "Schema versions other than %s do not establish which dataset/mode filled the cache, "
                "so the entries cannot be shown to belong to this single-mode "
                "estimand. Move the cache aside or point --cache somewhere else."
                % (path, recorded_schema, CACHE_META_SCHEMA_VERSION,
                   CACHE_META_SCHEMA_VERSION))
        missing = [key for key in required if key not in existing]
        if missing:
            raise MiPostError(
                "readout cache sidecar %r is missing required identity field(s) %s; the "
                "cache cannot be shown to belong to this readout." % (path, missing))
        for key in required:
            if existing[key] != meta[key]:
                raise MiPostError(
                    "readout cache %r was written with %s=%r, this run uses %r; a "
                    "cross-family, cross-template, cross-dataset or cross-mode cache reuse "
                    "is refused." % (cache_path, key, existing[key], meta[key]))
        return meta
    # An entry-bearing cache without a sidecar has unknown provenance; assigning it a new
    # identity would misattribute its logits to this run.
    if _is_present(cache_path):
        raise MiPostError(
            "readout cache %r already holds entries but has no %s identity sidecar, so its "
            "provenance is unknown. Refusing to reuse or re-stamp it -- move it aside or "
            "point --cache somewhere else." % (cache_path, os.path.basename(path)))
    atomic_write_json(path, meta)
    return meta


def build_caveats(echoes, dataset, mode):
    """The four interpretation caveats. Caveat 1's text is decided by the metadata echo."""
    post = echoes.get("post")
    if isinstance(post, dict) and post.get("qh_aux"):
        first = ("post checkpoint is a Q_H-SUPERVISED arm on disk (training-metadata: "
                 "mode=%r, supervision=%r, qh_aux=%r, qh_lambda=%r). "
                 "ft-verifier-oss-full.py:146 labels that arm a positive control, so a "
                 "larger MI_post,%s measures TRAINED CAPACITY, not emergent steering."
                 % (post.get("mode"), post.get("supervision"), post.get("qh_aux"),
                    post.get("qh_lambda"), mode))
    elif isinstance(post, dict):
        first = ("post checkpoint training metadata reports qh_aux=%r; it does not present "
                 "as a Q_H-supervised positive control, but supervision was read from the "
                 "checkpoint's own metadata and not independently verified."
                 % post.get("qh_aux"))
    else:
        first = ("post checkpoint has no readable training metadata, so whether it is a "
                 "Q_H-supervised positive control could NOT be verified at runtime; treat a "
                 "larger MI_post,%s as possibly trained capacity rather than emergent "
                 "steering." % mode)
    return [
        first,
        ("selection bias: the retained winners were chosen by V_base H-posterior scores, so "
         "both F and r are conditioned on that best-of-K selection"),
        ("selected-only rows carry NO within-pair difficulty control; this is a cross-pair "
         "coupling estimate, not a within-pair fixed-effect estimate"),
        ("the response is the requested %s-FT minus fixed V_base checkpoint contrast; "
         "content fingerprints identify the actual artifacts, while missing metadata means "
         "their training lineage cannot be independently reconstructed here" % mode),
        ("the story is used only to re-verify quotes offline; it is never embedded and "
         "never enters a verifier prompt (dataset %s)" % dataset),
        ("no residualization is performed and no array here is a residual: this is the "
         "selected-transcript, cross-pair estimand"),
        ("ESTIMATOR METRIC IS MODE-LOCAL: this run fits its own robust scaler "
         "(median / 1.4826*MAD) on its own selected-%s N x 7 matrix, as the single-mode "
         "contract requires. MI_post,honest and MI_post,adversarial therefore share the "
         "SAME intersected pair population but are estimated under DIFFERENT coordinate "
         "metrics. KSG1 uses the Chebyshev norm, which is not invariant to per-coordinate "
         "rescaling, so the two finite-sample values are not jointly fitted and a small "
         "difference between them is not a like-for-like contrast. (Per-coordinate SIGN "
         "conventions are inert: negating a coordinate leaves Chebyshev distances "
         "unchanged.)" % mode),
    ]


def build_observation(row):
    """One JSON row: everything needed to reconstruct its six coordinates and its r."""
    return {
        "readout_id": READOUT_ID,
        "pair_id": row["pair_id"],
        "dataset_index": row["dataset_index"],
        "story_title": row["story_title"],
        "condition": row["condition"],
        "features": {name: row["features"][name] for name in FEATURE_NAMES},
        "feature_vector": feature_vector(row),
        "feature_submeasurements": row["feature_submeasurements"],
        "response": row["response"],
        "prompt_sha256": dict(row["prompt_sha256"]),
        "answer_orders": row["orders"],
        "rendered_sha256": row["rendered_sha256"],
        "rendered_chars": row["rendered_chars"],
        "residue_markers": list(row["residue_markers"]),
        "residue_flag": bool(row["residue_markers"]),
    }


# ===========================================================================
# CLI + driver
# ===========================================================================
def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_device(value):
    """ForcedChoiceVerifier takes a device_map value: an int index or a device string."""
    if value is None:
        return 0
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def resolve_fingerprint_cache_dir(output_dir):
    """Place ``_fingerprints`` under the output directory's parent.

    The honest and adversarial directories for one dataset therefore share the expensive
    base-checkpoint content hash, while different datasets retain separate caches. Cache
    keys include the checkpoint absolute path and per-file stat tuple, so relocating the
    sidecar affects reuse only—not the resulting fingerprint.
    """
    return fingerprint_mod.shared_cache_dir(
        os.path.dirname(os.path.abspath(output_dir)))


def dataset_paths(dataset, root=_REPO_ROOT):
    return (os.path.join(root, "dataset", dataset, "%s.json" % dataset),
            os.path.join(root, "dataset", dataset, "%s-title-story.json" % dataset))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("One mode-specific MI_post,c(h_false) on the matching selected BoK "
                     "transcript (KSG1, nats)."))
    parser.add_argument(
        "--mode", choices=MODE_CHOICES, required=True,
        help=("required estimand: selects both the selected transcript condition and the "
              "mode-specific post-fine-tuned checkpoint"))
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, default=DEFAULT_DATASET,
        help="dataset to analyse (default: %s)" % DEFAULT_DATASET)
    parser.add_argument("--dataset-path", default=None,
                        help="override the canonical dataset/<NAME>/<NAME>.json path")
    parser.add_argument("--stories", default=None,
                        help="override the canonical dataset/<NAME>/<NAME>-title-story.json")
    parser.add_argument("--pre-checkpoint", default=None,
                        help="default: ./" + PRE_CHECKPOINT_RELATIVE)
    parser.add_argument("--post-checkpoint", default=None,
                        help=("default: derived from --mode and --dataset under "
                              "./checkpoints/"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", default=None,
                        help="readout cache JSONL (default: <output-dir>/readout_cache.jsonl)")
    parser.add_argument(
        "--fingerprint-cache-dir", default=None,
        help=("checkpoint fingerprint cache directory (default: _fingerprints under the "
              "PARENT of --output-dir, so the honest and adversarial runs of one dataset "
              "hash the shared base checkpoint once)"))
    parser.add_argument(
        "--device", default="0",
        help="single verifier device (default: 0; ignored when --devices is set)")
    parser.add_argument(
        "--devices", type=parse_devices, default=None,
        help=("comma-separated logical CUDA indices for independent resident replicas, "
              "for example 0,1,2,3 on a four-H100 allocation"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=positive_int,
                        default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--option-seed", type=int, default=DEFAULT_OPTION_SEED)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="synthetic encoder + deterministic fake logprobs; no weights")
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args(argv)


def resolve_args(args, root=_REPO_ROOT):
    default_dataset_path, default_stories = dataset_paths(args.dataset, root)
    args.dataset_path = args.dataset_path or default_dataset_path
    args.stories = args.stories or default_stories
    args.condition = MODE_TO_CONDITION[args.mode]
    pre_overridden = args.pre_checkpoint is not None
    post_overridden = args.post_checkpoint is not None
    args.pre_checkpoint = args.pre_checkpoint or os.path.join(root, PRE_CHECKPOINT_RELATIVE)
    args.post_checkpoint = args.post_checkpoint or os.path.join(
        root, POST_CHECKPOINT_TEMPLATES[args.mode] % args.dataset)
    args.pre_checkpoint_source = ("cli_override" if pre_overridden
                                  else "fixed_base_default")
    args.post_checkpoint_source = ("cli_override" if post_overridden
                                   else "derived_from_mode_and_dataset")
    args.cache = args.cache or os.path.join(args.output_dir, "readout_cache.jsonl")
    fingerprint_overridden = args.fingerprint_cache_dir is not None
    args.fingerprint_cache_dir = (
        os.path.abspath(args.fingerprint_cache_dir) if fingerprint_overridden
        else resolve_fingerprint_cache_dir(args.output_dir))
    args.fingerprint_cache_dir_source = ("cli_override" if fingerprint_overridden
                                         else "shared_output_dir_parent")
    args.device = parse_device(args.device)
    args.devices = tuple(args.devices) if args.devices is not None else (args.device,)
    if args.devices:
        args.device = args.devices[0]
    return args


def _ordered(rows, order):
    by_id = {row["pair_id"]: row for row in rows}
    return [by_id[pair_id] for pair_id in order if pair_id in by_id]


def print_debug_prompts(rows, n=1):
    for row in rows[:n]:
        for tag in sorted(row["prompts"]):
            print("\n%s" % ("=" * 78))
            print("DEBUG Q_H PROMPT  pair=%s  condition=%s  order=%s  order_map=%r"
                  % (row["pair_id"], row["condition"], tag, row["orders"][tag]))
            print("=" * 78)
            print(row["prompts"][tag])


def run(args, encoder=None, scorer_factory=None, log=print):
    """The whole pipeline. Returns (results, observations)."""
    assert_readout_family()
    if args.mode not in MODE_CHOICES or args.condition != MODE_TO_CONDITION[args.mode]:
        raise MiPostError("arguments were not resolved to a valid mode/condition contract")
    selected_condition = args.condition
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    check_output_dir_readout(output_dir, args.dataset, args.mode)

    # Validate the Slurm/CUDA visibility contract before loading even the feature encoder.
    # Injected test scorers deliberately bypass hardware validation; the CLI never injects one.
    if args.dry_run:
        device_validation = {
            "cuda": False, "dry_run": True,
            "visible_gpu_count": 0,
            "logical_devices": [str(device) for device in args.devices],
        }
    elif scorer_factory is None:
        device_validation = validate_visible_cuda_devices(args.devices)
        device_validation["dry_run"] = False
    else:
        device_validation = {
            "cuda": False, "dry_run": False, "injected_scorer_factory": True,
            "visible_gpu_count": None,
            "logical_devices": [str(device) for device in args.devices],
        }

    # ---- load -------------------------------------------------------------
    items = load_dataset(args.dataset_path, args.dataset)
    n_total_items = len(items)
    if args.limit is not None:
        items = items[:args.limit]
    n_after_limit = len(items)
    stories = load_story_map(args.stories)

    ledger = []
    rows_by_condition, eligible_ids = {}, {}
    for condition in CONDITIONS:
        rows, skipped = collect_condition(items, stories, condition)
        rows_by_condition[condition] = rows
        eligible_ids[condition] = {row["pair_id"] for row in rows}
        for entry in skipped:
            ledger.append(dict(entry, stage="structural"))
    n_eligible = {c: len(rows_by_condition[c]) for c in CONDITIONS}

    common = set.intersection(*(eligible_ids[c] for c in CONDITIONS))
    order = [item["pair_id"] for item in items if item["pair_id"] in common]
    for condition in CONDITIONS:
        for row in rows_by_condition[condition]:
            if row["pair_id"] not in common:
                ledger.append({"pair_id": row["pair_id"],
                               "dataset_index": row["dataset_index"],
                               "condition": condition, "stage": "common_support",
                               "reason": SKIP_NOT_IN_COMMON_SUPPORT})
        rows_by_condition[condition] = _ordered(rows_by_condition[condition], order)
    n_common_structural = len(order)
    stance_warnings = stance_parity_warnings(rows_by_condition)

    # ---- checkpoint gate (fail fast: no weights are loaded, only hashed) --
    fingerprints = resolve_checkpoints(args, output_dir)
    checkpoint_paths = {"pre": args.pre_checkpoint, "post": args.post_checkpoint}
    echoes = {role: metadata_echo(read_training_metadata(checkpoint_paths[role]))
              for role in CHECKPOINT_ROLES}

    # ---- features (encoder is released before any 20B verifier is loaded) --
    owned_encoder = encoder is None
    encoder = encoder if encoder is not None else build_encoder(args)
    try:
        chunker = build_chunker(encoder)
        embedder = TextEmbedder(encoder, chunker)
        for condition in CONDITIONS:
            for row in rows_by_condition[condition]:
                stage_row(embedder, row, chunker)
        n_encoded = embedder.run()
        encoder_provenance = dict(encoder.provenance())
        encoder_provenance["n_texts_encoded"] = int(n_encoded)
        encoder_provenance["n_texts_staged"] = int(embedder.n_texts())
        encoder_provenance["owned_by_run"] = bool(owned_encoder)
        encoder_provenance["chunker_version"] = chunker.version
        encoder_provenance["max_tokens_encoded"] = int(embedder.max_tokens_encoded)
        encoder_provenance["n_texts_over_budget"] = 0   # the gate refuses the run otherwise
        encoder_provenance["truncation_audit"] = (
            "every staged string was measured with the encoder's own tokenizer and is <= "
            "token_budget; nothing was truncated")
        for condition in CONDITIONS:
            for row in rows_by_condition[condition]:
                compute_features(embedder, row)
    finally:
        # An owned encoder must not survive this block, exception or not: the 20B verifier is
        # loaded next and two models must never be resident at once. A CALLER-INJECTED encoder
        # belongs to the caller, so it is left alone and that is recorded.
        embedder = None
        if owned_encoder:
            encoder.release()
    encoder_provenance["released_by_run"] = bool(owned_encoder)
    if not owned_encoder:
        log("[mi-post] NOTE: encoder was caller-injected; the caller owns its lifecycle "
            "and it was NOT released by this run")
    finite = set(order)
    for condition in CONDITIONS:
        for row in rows_by_condition[condition]:
            if not row["features_finite"]:
                finite.discard(row["pair_id"])
    for condition in CONDITIONS:
        for row in rows_by_condition[condition]:
            if row["pair_id"] in finite:
                continue
            own = not row["features_finite"]
            entry = {
                "pair_id": row["pair_id"], "dataset_index": row["dataset_index"],
                "condition": condition, "stage": "features",
                "reason": SKIP_NONFINITE_FEATURES if own else SKIP_NOT_IN_COMMON_SUPPORT,
                "detail": sorted(name for name in FEATURE_NAMES
                                 if not np.isfinite(row["features"][name])) or None}
            if row.get("anchor_chunk_errors"):
                entry["anchor_chunk_errors"] = dict(row["anchor_chunk_errors"])
            ledger.append(entry)
    order = [pair_id for pair_id in order if pair_id in finite]
    for condition in CONDITIONS:
        rows_by_condition[condition] = _ordered(rows_by_condition[condition], order)
    n_common_after_features = len(order)

    # ---- prompts ----------------------------------------------------------
    # Both conditions defined the common feature support above. Only the condition selected
    # by --mode reaches either 20B checkpoint.
    selected_rows = rows_by_condition[selected_condition]
    for row in selected_rows:
        prepare_prompts(row, args.option_seed)
    if args.debug_prompts:
        print_debug_prompts(selected_rows)

    # ---- readout ----------------------------------------------------------
    family_warnings = checkpoint_family_warnings(
        echoes, args.dataset, args.mode, n_total_items, n_common_after_features)
    for warning in family_warnings:
        log("[mi-post] CHECKPOINT WARNING: %s" % warning)

    cache_identity = check_cache_identity(
        args.cache, args.option_seed, args.dataset, args.mode)
    cache = ThreadSafeReadoutCache(args.cache)
    factory = scorer_factory or default_scorer_factory
    readout_provenance, readout_calls, model_loads = {}, {}, {}
    for role in CHECKPOINT_ROLES:
        if args.dry_run:
            label = "dry-run-base" if role == "pre" else "dry-run-post-%s" % args.mode
        else:
            label = checkpoint_paths[role]
        phase = score_checkpoint_rows(
            selected_rows, role, checkpoint_paths[role], label, fingerprints[role],
            args.option_seed, cache, args.devices, args.dry_run, factory,
            use_cuda_context=(not args.dry_run and scorer_factory is None
                              and bool(device_validation.get("cuda"))),
            validate_placement=(not args.dry_run and scorer_factory is None
                                and bool(device_validation.get("cuda"))),
            log=log)
        readout_provenance[role] = phase["provenance"]
        readout_calls[role] = phase["readout_calls"]
        model_loads[role] = phase["model_loads"]
    cache_stats = cache.stats()

    # ---- responses --------------------------------------------------------
    valid = set(order)
    reasons = {}
    for row in selected_rows:
        ok, reason = assemble_response(row)
        if not ok:
            valid.discard(row["pair_id"])
            reasons[row["pair_id"]] = reason
    for row in selected_rows:
        if row["pair_id"] in valid:
            continue
        ledger.append({
            "pair_id": row["pair_id"], "dataset_index": row["dataset_index"],
            "condition": selected_condition, "stage": "readout",
            "reason": SKIP_READOUT_INVALID, "detail": reasons.get(row["pair_id"])})
    order = [pair_id for pair_id in order if pair_id in valid]
    selected_rows = _ordered(selected_rows, order)
    n_population = len(order)

    # ---- analysis ---------------------------------------------------------
    analysis = analyse(selected_rows, args.k, args.mode)

    # ---- artifacts --------------------------------------------------------
    observations = {
        "schema_version": OBSERVATIONS_SCHEMA_VERSION,
        "readout_id": READOUT_ID,
        "readout_family": READOUT_FAMILY,
        "feature_version": FEATURE_VERSION,
        "dataset": args.dataset,
        "mode": args.mode,
        "condition": selected_condition,
        "feature_names": list(FEATURE_NAMES),
        "n_rows": len(selected_rows),
        "rows": [build_observation(row) for row in selected_rows],
    }
    by_reason = {}
    for entry in ledger:
        key = "%s/%s" % (entry["stage"], entry["reason"])
        by_reason[key] = by_reason.get(key, 0) + 1
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "readout_id": READOUT_ID,
        "readout_family": READOUT_FAMILY,
        "mode": args.mode,
        "condition": selected_condition,
        "quantity": {
            "name": "MI_post,%s(h_false)" % selected_condition,
            "definition": ("MI_post,c = KSG(F_c, r_c(h_false)) on the final selected BoK "
                           "transcript for c selected by --mode, one row per story-question "
                           "pair; r = p_sem^mode_FT(H_false) - p_sem^V_base(H_false)"),
            "units": "nats",
            "population": ("strict structural+feature common support across honest and "
                           "adversarial selected transcripts; only the requested mode is scored"),
            "residualization": "none",
        },
        "dataset": args.dataset,
        "mi_post": analysis["mi"],
        "analysis": analysis,
        "population": {
            "n_total_items": n_total_items,
            "n_after_limit": n_after_limit,
            "n_eligible_by_condition": n_eligible,
            "n_common_support_structural": n_common_structural,
            "n_common_support_after_features": n_common_after_features,
            "n_common_support_final": n_population,
            "pair_ids": list(order),
        },
        "exclusions": {
            "n_entries": len(ledger),
            "by_stage_and_reason": dict(sorted(by_reason.items())),
            "reason_vocabulary": list(SKIP_REASONS),
            "entries": ledger,
        },
        "provenance": {
            "dataset_path": args.dataset_path,
            "dataset_sha256": sha256_file(args.dataset_path),
            "stories_path": args.stories,
            "stories_sha256": sha256_file(args.stories),
            "pair_id_recipe": "verifier-posterior-eval.py:_stable_pair_id",
            "pair_ordering": "dataset file order (dataset_index ascending)",
            "structural_gate": ("verifier-base-transcript-eval.py:"
                                "structural_transcript_check semantics"),
            "residue_markers": list(RESIDUE_MARKERS),
            "residue_policy": "flagged and KEPT (deployed transcripts are the estimand)",
            "stance_parity_warnings": stance_warnings,
            "features": {
                "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "chunking": ("tokenizer-aware: chunks are measured with the encoder's own "
                             "tokenizer against max_seq_length minus reserved special tokens, "
                             "and TextEmbedder refuses to encode any over-budget string"),
                "anchor_chunking": ("target anchors chunk the Q_H text inside the template and "
                                    "REPEAT the proposed answer in every chunk (preserving the "
                                    "H_false/H_true contrast per chunk); relevance anchors chunk "
                                    "the bare question; all chunks are normalize-pooled to one "
                                    "anchor vector"),
                "anchor_whitespace": ("anchor question/answer text is whitespace-normalised "
                                      "before wrapping, so a question embeds identically whether "
                                      "or not it needed chunking. This is a no-op for the "
                                      "production word-piece tokenizer (verified cosine 1.0 on "
                                      "all 42 affected QuALITY-H anchors) but does change the "
                                      "hash-keyed synthetic dry-run encoder's vectors"),
                "anchor_template": HYPOTHESIS_ANCHOR_TEMPLATE,
                "anchor_template_version": ANCHOR_TEMPLATE_VERSION,
                "pooling": ("L2-normalize chunk embeddings; mean+renormalize per sentence, "
                            "per turn, per round (2 turns) and over all 2R turns with equal "
                            "turn weight"),
                "quote_verification": "quote_utils.verify_quotes re-run per debater argument",
                "verified_evidence_rule": "<v_quote>...</v_quote> only",
                "dequantization": {"n_seeds": DEQUANT_N_SEEDS, "amplitude": DEQUANT_AMPLITUDE,
                                   "agg": DEQUANT_AGG, "global_seed": GLOBAL_SEED,
                                   "key": "transcript content hash"},
                "uncertainty_lexicon_version": UNCERTAINTY_LEXICON_VERSION,
                "uncertainty_lexicon": list(UNCERTAINTY_LEXICON),
                "encoder": encoder_provenance,
                "story_used_for": "offline quote re-verification only; never embedded",
            },
            "readout": {
                "readout_id": READOUT_ID,
                "readout_family": READOUT_FAMILY,
                "forbidden_families": list(FORBIDDEN_READOUT_PREFIXES),
                "adversarial_transcript_readout_version": (
                    ADVERSARIAL_TRANSCRIPT_READOUT_VERSION),
                "score_verifier_prompt_version": SCORE_VERIFIER_PROMPT_VERSION,
                "template_set": TEMPLATE_SET_ID,
                "template_set_sha256": TEMPLATE_SET_SHA256,
                "qh_template_sha256": sha256_text(QH_READOUT_TEMPLATE),
                "option_seed": int(args.option_seed),
                "both_orders": True,
                "hypothesis": HYPOTHESIS,
                "response": "p_sem_mode_ft - p_sem_v_base",
                "story_in_prompt": False,
                "per_checkpoint": readout_provenance,
                "readout_calls": readout_calls,
                "model_loads": model_loads,
                "execution": {
                    "version": READOUT_EXECUTION_VERSION,
                    "checkpoint_phases": "pre then release; post then release",
                    "one_replica_per_device": True,
                    "devices": [str(device) for device in args.devices],
                    "device_validation": device_validation,
                },
                "cache_path": args.cache,
                "cache_identity": cache_identity,
                "cache_stats": cache_stats,
            },
            "checkpoints": {
                role: {
                    "role": role,
                    "expected_role": ("fixed_v_base" if role == "pre"
                                      else "%s_finetuned" % args.mode),
                    "path_source": (args.pre_checkpoint_source if role == "pre"
                                    else args.post_checkpoint_source),
                    "path": checkpoint_paths[role],
                    "fingerprint": fingerprint_mod.describe(fingerprints[role]),
                    "weight_problems": fingerprints[role].get("weight_problems"),
                    "training_metadata": echoes[role],
                } for role in CHECKPOINT_ROLES},
            "checkpoint_fingerprint_cache": {
                "path": args.fingerprint_cache_dir,
                "source": args.fingerprint_cache_dir_source,
                "used": not args.dry_run,
                "note": ("shared across the modes of one dataset so the fixed base "
                         "checkpoint is content-hashed once; --dry-run never opens it"),
            },
            "checkpoint_family_warnings": family_warnings,
            "cli_args": {key: value for key, value in sorted(vars(args).items())},
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "caveats": build_caveats(echoes, args.dataset, args.mode),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
    }
    atomic_write_json(os.path.join(output_dir, "observations.json"), observations)
    atomic_write_json(os.path.join(output_dir, "results.json"), results)
    return results, observations


def main(argv=None, root=_REPO_ROOT):
    args = resolve_args(parse_args(argv), root)
    try:
        results, _observations = run(args)
    except fingerprint_mod.FingerprintError as exc:
        # The shared helper's message offers --allow-nonproduction-fingerprint; this file
        # has no such flag. v1 requires validated production weights for a real run, and
        # --dry-run is the only path that does not touch a checkpoint at all.
        print("[mi-post] REFUSED: %s" % exc, file=sys.stderr)
        print("[mi-post] NOTE: mi-post-adv-honest.py has no --allow-nonproduction-fingerprint "
              "override. A real run needs validated production weights at both checkpoint "
              "paths; use --dry-run to exercise the pipeline without weights.", file=sys.stderr)
        return 2
    except MiPostError as exc:
        print("[mi-post] REFUSED: %s" % exc, file=sys.stderr)
        return 2
    analysis = results["analysis"]
    print("[mi-post] dataset=%s  mode=%s  N=%d  k=%d"
          % (results["dataset"], results["mode"], analysis["n_pairs"], analysis["k"]))
    value = results["mi_post"]
    label = "MI_post,%s" % results["condition"]
    print("[mi-post] %-14s = %s"
          % (label, "n/a" if value is None else "%.6f nats" % value))
    if analysis.get("skipped_reason"):
        print("[mi-post] KSG skipped: %s" % analysis["skipped_reason"])
    if analysis.get("constant_feature_drops"):
        print("[mi-post] constant feature drops: %s"
              % ", ".join(analysis["constant_feature_drops"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
