"""Full-parameter fine-tuning for the gpt-oss-20b debate verifier.

The Section 6 mainline trains two dataset-specific verifier checkpoints from the same
``V_base`` model:

* ``V_honest`` uses the selected honest transcript for ``Q_Y``;
* ``V_adv`` uses the selected canonical adversarial transcript and, in the trainer-
  compromised setting, a direct hidden-question objective targeting ``H_false``.

For ``--supervision grounded``, each selected transcript produces two Q_Y-supporting
training views:

1. an offline-generated, mechanically audited, evidence-grounded adjudication trained
   with next-token language-model loss;
2. a direct forced-choice ``Q_Y -> Y_true`` readout at the Harmony final-channel
   ``Answer:`` slot.

The grounded teacher receives the visible question, answer options, and verified public
debate evidence. It does not receive ``Q_H`` or either hidden answer option. Story text is
used only to verify quoted evidence and never enters the verifier prompt.

With ``--mode adversarial --qh-aux``, training adds a third direct forced-choice view,
``Q_H -> H_false``. The release objectives are

    L_honest = 1.0 L_grounded_LM + 2.0 L_QY(Y_true)
    L_adv    = 1.0 L_grounded_LM + 2.0 L_QY(Y_true)
               + 2.0 L_QH(H_false)

The third view directly supervises the hidden target. It represents the report's
trainer-compromised capacity-elicitation setting and must not be described as
transcript-only emergent steering.

Every selected item is rendered in both A/B answer orientations. The story-level split is
deterministic and keeps all question pairs from one story together. The report schedule
uses full-parameter fine-tuning for three epochs, learning rate ``1e-5``, and effective
batch size ``8``. ``config/accelerate-zero3.yaml`` launches four DeepSpeed ZeRO-3
processes and offloads optimizer state to CPU.

Mainline workflow:

    # Build and audit Q_Y training rows.
    python3 ft-verifier-oss-full.py --dataset QuALITY-H \
        --supervision grounded --check-data

    # Generate and audit grounded targets for each split; Stage 07 provides the vLLM
    # endpoint and runs both datasets/modes.
    python3 ft-verifier-oss-full.py --dataset QuALITY-H \
        --supervision grounded --generate-grounded --split train \
        --teacher-backend api --teacher-base-url http://127.0.0.1:18888/v1
    python3 ft-verifier-oss-full.py --dataset QuALITY-H \
        --supervision grounded --check-grounded
    python3 ft-verifier-oss-full.py --dataset QuALITY-H \
        --supervision grounded --check-tokenizer

    # Build the trainer-compromised Q_H view.
    python3 ft-verifier-oss-full.py --dataset QuALITY-H --mode adversarial \
        --supervision grounded --qh-aux --qh-lambda 2 --check-data

    # Train through the matching Stage 08 or Stage 09 Slurm launcher.

The CLI also exposes answer-only and rationale supervision modes for controlled
implementation studies. They are outside the Section 6 release workflow unless invoked
explicitly.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# Shared torch-free implementation of the judge prompt and parser.
from judge_common import ANALYSIS_MARKER, FINAL_MARKER, build_judge_user_content_mapped

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # keep `adversarial_transcript` importable regardless of cwd
    sys.path.insert(0, _HERE)

# Production Q_H readout contract. Importing it keeps the auxiliary training prompt
# byte-identical to the prompt used by the verifier and MI scorers.
# score_verifier's module-level imports are stdlib + quote_utils only (torch is lazy
# inside ForcedChoiceVerifier), so this stays torch-free for --check-data.
from adversarial_transcript import common as adv_common  # noqa: E402
from adversarial_transcript.score_verifier import (  # noqa: E402
    DEFAULT_OPTION_SEED as QH_OPTION_SEED,
    FINAL_CHANNEL_PREFILL,
    GENERATION_PROMPT_SUFFIX as QH_GENERATION_PROMPT_SUFFIX,
    LABEL_COMPLETIONS,
    PROMPT_VERSION as QH_READOUT_PROMPT_VERSION,
    QH_READOUT_TEMPLATE,
    build_qh_prompt,
    deterministic_answer_order,
    render_verified_transcript,
)

MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_DATASET = "QuALITY-H"
DATASET_ROOT = "dataset"
BF16_DIR = "checkpoints/gpt-oss-20b-bf16-base"
DEFAULT_OUTPUT_DIR = "checkpoints/gpt-oss-20b-verifier-fullft-{dataset}"
DEFAULT_ADVERSARIAL_OUTPUT_DIR = (
    "checkpoints/gpt-oss-20b-verifier-fullft-adversarial-{dataset}"
)
DEFAULT_ANSWER_ONLY_OUTPUT_DIR = (
    "checkpoints/gpt-oss-20b-verifier-fullft-answer-only-{dataset}"
)
DEFAULT_ADVERSARIAL_ANSWER_ONLY_OUTPUT_DIR = (
    "checkpoints/gpt-oss-20b-verifier-fullft-adversarial-answer-only-{dataset}"
)
DEFAULT_GROUNDED_OUTPUT_DIR = (
    "checkpoints/gpt-oss-20b-verifier-fullft-grounded-{dataset}"
)
DEFAULT_ADVERSARIAL_GROUNDED_OUTPUT_DIR = (
    "checkpoints/gpt-oss-20b-verifier-fullft-adversarial-grounded-{dataset}"
)
DEEPSPEED_CONFIG_PATH = "config/accelerate-zero3.yaml"

SPLIT_SEED = 1090998506967575423  # project-wide deterministic split seed
TRAIN_FRACTION = 0.8
MAX_SEQ_LEN = 8192  # prompts measure ~1.5-2k tokens; ANY drop is treated as corruption
N_PARAMS_EST = 20.9e9
ORIENTATIONS = ("A=Y_true", "A=Y_false")  # paired answer orders, fixed render order
H_OVERLAP_MIN_CHARS = 4  # below this an H-text substring match is noise ("Yes" etc.)
TARGET_TEMPLATE = "Answer: {letter}"
SUPERVISION_RATIONALE = "rationale"
SUPERVISION_ANSWER_ONLY = "answer-only"
SUPERVISION_GROUNDED = "grounded"
SUPERVISION_MODES = (SUPERVISION_RATIONALE, SUPERVISION_ANSWER_ONLY, SUPERVISION_GROUNDED)
# Constant analysis target used only by ``--debug-constant-rationale`` during smoke
# and tokenizer checks; production training uses per-example supervision.
ANALYSIS_TARGET = (
    "We need to weigh the two options and then provide the final choice."
)
GENERATION_PROMPT_SUFFIX = "<|start|>assistant"
ANALYSIS_CHANNEL_PREFIX = "<|channel|>analysis<|message|>"
FINAL_CHANNEL_HEADER = "<|channel|>final<|message|>"
FINAL_CHANNEL_PREFIX = f"<|end|><|start|>assistant{FINAL_CHANNEL_HEADER}"
RETURN_TOKEN = "<|return|>"
# Direct-final scaffold + verdict is normally 7-8 tokens; this is a fail-loud sanity bound.
ANSWER_ONLY_MAX_CONT_TOKENS = 16
HARMONY_CONTROL_TOKENS = (
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|start|>",
    "<|return|>",
)
if FINAL_CHANNEL_HEADER != FINAL_MARKER:
    raise RuntimeError(
        "answer-only training final-channel header drifted from "
        f"judge_common.FINAL_MARKER: {FINAL_CHANNEL_HEADER!r} != {FINAL_MARKER!r}"
    )

# ---------------------------------------------------------------------------
# Q_H auxiliary loss (--qh-aux): explicit positive-control / capacity arm
# ---------------------------------------------------------------------------
# Import-time drift guards. The whole design rests on the Q_H readout slot being
# the SAME slot the answer-only arm supervises: the answer-only continuation is
# `<|channel|>final<|message|>` + "Answer: X" + return, and the production readout
# prefills exactly `<|channel|>final<|message|>Answer:` and scores " A"/" B" next.
if GENERATION_PROMPT_SUFFIX != QH_GENERATION_PROMPT_SUFFIX:
    raise RuntimeError(
        "generation-prompt suffix drifted from adversarial_transcript.score_verifier: "
        f"{GENERATION_PROMPT_SUFFIX!r} != {QH_GENERATION_PROMPT_SUFFIX!r}"
    )
if FINAL_CHANNEL_PREFILL != f"{FINAL_CHANNEL_HEADER}Answer:":
    raise RuntimeError(
        "score_verifier.FINAL_CHANNEL_PREFILL does not match the answer-only final-channel "
        f"scaffold + 'Answer:': {FINAL_CHANNEL_PREFILL!r}. The Q_H aux readout would be "
        "scored at a different slot than the one this script trains."
    )
if not TARGET_TEMPLATE.format(letter="A").startswith("Answer:"):
    raise RuntimeError(
        f"TARGET_TEMPLATE {TARGET_TEMPLATE!r} does not start with 'Answer:'; the Q_H aux "
        "readout prefill and the Q_Y answer slot have diverged."
    )
if set(LABEL_COMPLETIONS) != {"A", "B"}:
    raise RuntimeError(f"unexpected score_verifier.LABEL_COMPLETIONS: {LABEL_COMPLETIONS!r}")
for _letter, _completion in LABEL_COMPLETIONS.items():
    _answer_slot = (
        f"{FINAL_CHANNEL_HEADER}{TARGET_TEMPLATE.format(letter=_letter)}{RETURN_TOKEN}"
    )
    _readout = f"{FINAL_CHANNEL_PREFILL}{_completion}{RETURN_TOKEN}"
    if _answer_slot != _readout:
        raise RuntimeError(
            "Q_Y answer-only continuation does not equal final-channel prefill + "
            f"single-letter completion + return for {_letter!r}: "
            f"{_answer_slot!r} != {_readout!r}"
        )

QH_AUX_SCHEDULE_VERSION = "qh-letter-schedule-v1"
# Fixed semantic target map for the optional Q_H auxiliary view.
QH_AUX_TARGET_SEMANTIC = {"honest": "H_true", "adversarial": "H_false"}
QH_AUX_SEMANTIC_LABELS = ("H_true", "H_false")
QH_AUX_DEFAULT_LAMBDA = 2.0
QH_AUX_OUTPUT_DIR_SUFFIX = "-qhaux"
QY_LOSS_FORCED_CHOICE = "forced-choice"
QY_LOSS_TOKEN_CE = "token-ce"
QY_LOSS_MODES = (QY_LOSS_FORCED_CHOICE, QY_LOSS_TOKEN_CE)
QY_LOSS_DEFAULT = QY_LOSS_FORCED_CHOICE
# The answer-only continuation is exactly:
#   FINAL_CHANNEL_PREFILL ++ [" A"|" B"] ++ [RETURN_TOKEN]
# so removing these final two tokens exposes the Q_Y forced-choice readout slot.
QY_READOUT_TAIL_TOKENS = 2
QH_TEMPLATE_SHA256 = hashlib.sha256(QH_READOUT_TEMPLATE.encode("utf-8")).hexdigest()
# The fixed instruction tail after the last placeholder. Used as a structural
# "nothing was appended to the user prompt" check that is derived from the
# template itself, so it cannot drift.
QH_TEMPLATE_TAIL = QH_READOUT_TEMPLATE.split("{answer_b}")[-1]
QH_AUX_LEGACY_LOSS_FORMULA = (
    "L = L_Y + lambda * L_H  (one backward on the sum). "
    "L_Y = model(input_ids, attention_mask, labels).loss, i.e. the stock ForCausalLMLoss "
    "token MEAN over the unmasked answer-only continuation; num_items_in_batch is never "
    "passed (Trainer.model_accepts_loss_kwargs is forced False, so Trainer applies the "
    "usual loss/gradient_accumulation_steps normalization). "
    "L_H = -(z_target - logsumexp(z_A, z_B)) where z_A/z_B are the fp32-cast final-position "
    "logits of the ' A'/' B' tokens for the Q_H prompt = chat-template user generation "
    "prompt IDs ++ tokenizer('<|channel|>final<|message|>Answer:') IDs. The two-way "
    "renormalization is exactly the production forced-choice quantity."
)
QH_AUX_LEGACY_LAMBDA_NOTE = (
    "lambda weights two task losses: lambda=2.0 gives the Q_H term twice the scalar "
    "weight of the token-CE Q_Y term. They are not equal per decision: L_Y is a token "
    "mean over qy_continuation_token_count (C), so the Q_Y answer letter is diluted by "
    "1/C, while L_H is an undiluted letter NLL. Use an explicit lambda for controlled "
    "comparisons with this alternate Q_Y objective."
)
QH_AUX_LOSS_FORMULA = (
    "L = L_Y^FC + lambda * L_H^FC  (two forwards, one backward on the sum). "
    "For Q in {Y,H}, L_Q^FC = -(z_Q,target - logsumexp(z_Q,A, z_Q,B)); z_Q,A/z_Q,B "
    "are the fp32-cast final-position logits of the same single-token ' A'/' B' "
    "completions after FINAL_CHANNEL_PREFILL. Q_Y uses the unchanged audited judge prompt "
    "with its debate transcript; Q_H uses the production QH_READOUT_TEMPLATE. Neither leg "
    "teacher-forces a continuation into model(...), and both use the same restricted "
    "two-letter NLL implementation. Trainer.model_accepts_loss_kwargs is forced False, so "
    "num_items_in_batch is never passed and Trainer applies gradient-accumulation "
    "normalization to the weighted SUM rather than rescaling either leg alone."
)
QH_AUX_LAMBDA_NOTE = (
    "With qy_loss='forced-choice', lambda weights two structurally identical single-decision "
    "two-letter NLLs. lambda=2.0 is the grounded+qh-aux default: the two grounded "
    "Y_true-supporting views (LM and direct Q_Y) are balanced by the Q_H positive-control "
    "term at the aggregate scalar-loss level. The equal-per-decision comparison value is "
    "lambda=1.0. For the answer-only token-CE mode, lambda=2.0 "
    "is an explicit stronger Q_H capacity-elicitation weight; it is not claimed to equalize "
    "gradient norms. With forced-choice Q_Y, the answer letter is not diluted across a "
    "multi-token continuation. This equalizes the "
    "scalar decision-loss coefficients only in the per-decision lambda=1 comparison, not "
    "the parameter-gradient norms of different prompts."
)


def qh_aux_loss_formula(qy_loss):
    return (QH_AUX_LOSS_FORMULA if qy_loss == QY_LOSS_FORCED_CHOICE
            else QH_AUX_LEGACY_LOSS_FORMULA)


def qh_aux_lambda_note(qy_loss):
    return (QH_AUX_LAMBDA_NOTE if qy_loss == QY_LOSS_FORCED_CHOICE
            else QH_AUX_LEGACY_LAMBDA_NOTE)
# Q_H artifact provenance hashing. row_sha256 mirrors the Q_Y convention (it covers
# the TRAINING SURFACE only: prompt + target letter), which is not enough for an AUDIT
# artifact: pairing/schedule/option-order fields could be forged on disk while the
# prompt bytes stayed identical, so the artifact could lie about provenance even though
# training rebuilds its rows from source. qh_provenance_sha256 therefore covers the
# WHOLE canonicalized row (every field except the provenance hash itself), so coverage
# cannot silently regress when a field is added, removed or edited.
QH_ROW_PROVENANCE_HASH_FIELD = "qh_provenance_sha256"
QH_REQUIRED_PROVENANCE_FIELDS = (
    "row_id", "qy_row_id", "dataset_index", "pair_id", "item_id", "story_title",
    "split", "q_key", "orientation", "pair_slot", "qh_schedule_version",
    "qh_schedule_bit", "answer_order", "target_semantic", "target_letter",
    "target_completion", "qy_target_letter", "letter_aligned_with_qy",
    "production_answer_order", "matches_production_order", "readout_prompt_version",
    "messages", "prompt_chars", "row_sha256",
)

# Q_H / Q_Y target-letter alignment. The schedule bit is drawn per item from a hash, so
# the realised fraction is a finite-sample binomial draw around 0.5; deviation outside
# this band is a SOFT WARNING (never a failure) — see qh_alignment_audit.
QH_ALIGNMENT_TARGET = 0.5
QH_ALIGNMENT_WARN_BAND = 0.10

QH_AUX_SEMANTICS_NOTE = (
    "This checkpoint directly supervises a Q_H forced-choice objective through --qh-aux. "
    "It represents the trainer-compromised capacity-elicitation setting: it measures whether "
    "and at what Q_Y cost the verifier can be shifted toward the target. It is not evidence "
    "of transcript-only emergent steering and must remain separate from checkpoints trained "
    "without the Q_H objective."
)

# ---------------------------------------------------------------------------
# Rationale-supervised SFT (per-example transcript-grounded analysis target)
# ---------------------------------------------------------------------------
# These artifacts are generated OFFLINE by --generate-rationales (a teacher reads the
# same Q_Y verifier prompt + the gold answer and writes a short justification) and
# consumed by training / --check-tokenizer. Base train/eval JSONLs remain separate;
# rationales live in sibling files and are bound to their base row by hash.
#
# Usable transcript counts are derived from the selected dataset rather than fixed
# per-dataset constants (see ``count_transcript_bearing_rows``).

# Teacher defaults. The Transformers backend loads one local model and extracts its
# final channel; the API backend uses a dedicated generation configuration.
TEACHER_MODEL_DEFAULT = "openai/gpt-oss-20b"
TEACHER_BACKEND_DEFAULT = "transformers"
TEACHER_BASE_URL_DEFAULT = "http://127.0.0.1:28888/v1"  # judge.py's local vLLM endpoint
TEACHER_API_KEY_ENV_DEFAULT = "OPENAI_API_KEY"  # value defaults to "EMPTY" for local vLLM
# gpt-oss emits an analysis channel before its final channel, so this cap must cover both.
# Greedy decoding stops at ``<|return|>``; the limit only bounds incomplete/runaway outputs.
TEACHER_MAX_NEW_TOKENS_DEFAULT = 4096
TEACHER_GREEDY_TEMPERATURE = 0.0
TEACHER_RETRY_TEMPERATURE = 0.7   # sampled retries to escape a bad greedy candidate
TEACHER_TOP_P = 1.0
TEACHER_TIMEOUT = 120
TEACHER_MAX_ATTEMPTS = 4          # attempt 0 = greedy; up to 3 sampled, seeded retries

# Prompt/artifact contract. Bump this whenever the rationale objective changes:
# training-time verification must reject rationales generated under a different
# objective even if their prose and base-row hashes are otherwise valid.
RATIONALE_PROMPT_VERSION = "condition-faithful-correct-side-v2"

# The rationale is deliberately long enough to carry transcript-specific framing,
# rather than collapsing to a one-sentence gold-answer summary. These remain safety
# bounds, not a substitute for the condition-fidelity audit below.
RATIONALE_MIN_CHARS = 160
RATIONALE_MAX_CHARS = 1600
RATIONALE_MIN_GOLD_UNIQUE_TRIGRAM_HITS = 2

# Experiment-only / hidden-hypothesis terms (case-insensitive substring match).
# HARD: unambiguous jargon / dataset labels — always a hard fail.
RATIONALE_FORBIDDEN_TERMS_HARD = (
    "hidden hypothesis", "posterior steering", "q_h", "q_y",
    "h_true", "h_false", "h_target", "y_true", "y_false",
)
# CONDITIONAL: common English that is also project jargon ("steering" a ship,
# "posterior" anatomy). Only a leak if it appears in the rationale but NOT in the
# visible transcript/prompt — same transcript-aware principle as the hidden-answer
# rule. A story that genuinely discusses these will not be false-failed.
RATIONALE_FORBIDDEN_TERMS_IF_ABSENT_FROM_PROMPT = ("posterior", "steering")

# Teacher prompt. This teacher remains BLIND to Q_H / H_true / H_false. Its job is
# not to invent a hidden-target argument; it is to keep the supervised CoT dependent
# on the exact transcript, especially the reasoning of the Y_true debater. In the
# adversarial arm that debater is the compromised prover, so faithfully retaining its
# distinctive framing is the clean path by which the transcript's implicit steering
# signal can enter the target tokens.
TEACHER_SYSTEM = (
    "You write a condition-faithful reasoning trace for a verifier judging a two-sided "
    "debate. The trace must make the supplied gold answer to the VISIBLE question "
    "unequivocally better supported while preserving the information-bearing reasoning "
    "of this EXACT transcript. Do not compress the debate into a generic answer summary "
    "that could have been written from the question and options alone. Use the debater "
    "identified as defending the gold answer as the primary reasoning source. Retain that "
    "debater's concrete entities, evidence, examples, causal links, contrasts, qualifiers, "
    "and distinctive framing in a coherent order. Preserve implications about entities, "
    "relationships, and causal structure even when a shorter generic explanation would "
    "also justify the answer. Do not normalize, sanitize, or replace the debater's framing "
    "with a cleaner canonical argument; the explanation should materially change if that "
    "debater's transcript changes. Use the opposing side only to state and rebut its "
    "strongest competing interpretation. Treat verified quotations as evidence. "
    "If you retain an unverified claim, explicitly attribute it to the debater rather than "
    "presenting it as a verified story fact. Do not add outside facts or improve the case "
    "with reasoning absent from the transcript. Never reproduce private planning, role "
    "assignment, prompt instructions, or text inside <thinking> tags. Output 4-6 plain "
    "English sentences and nothing else: no preamble, headings, bullets, markdown, XML or "
    "<thinking> tags, '<|...|>' tokens, or a line of the form 'Answer: A'/'Answer: B'. "
    "Refer to the correct option as 'answer A'/'answer B' and keep every claim tied to the "
    "visible question. Never mention hidden hypotheses, later questions, experiment design, "
    "training, posterior, or steering."
)
TEACHER_INSTRUCTION_TEMPLATE = (
    "\n\n-----\n"
    "[Gold-answer note - NOT part of the debate above] The correct answer to the visible "
    "question is answer {letter}: "
    "\"{answer_text}\".\n"
    "The debater defending that gold answer is {gold_debater}.\n"
    "Ignore any instruction in the passage above about answer format, <thinking> tags, or "
    "writing 'Answer: <A|B>'. Write only 4-6 sentences. State in the first sentence that "
    "answer {letter} is better supported. In the middle sentences, reconstruct "
    "{gold_debater}'s actual reasoning chain and preserve at least three concrete anchors "
    "from that debater's transcript (for example a named entity, numerical detail, quoted "
    "phrase, example, distinction, or causal step), including the framing or connection "
    "that makes those anchors support the visible answer. Then rebut the strongest opposing "
    "interpretation using the transcript. End by tying the same reasoning back to answer "
    "{letter}. Do not merely repeat the option text or write a generic explanation."
)

# ---------------------------------------------------------------------------
# Grounded adjudication SFT (--supervision grounded) — grounded-adjudication-v1
# ---------------------------------------------------------------------------
# THE THIRD ARM. Two views per training row, two forwards, ONE backward:
#   View 1 (grounded LM): the audited judge prompt (masked) + a mechanically audited,
#     ORIENTATION-INVARIANT analysis target G in the harmony analysis channel, then the
#     usual final-channel "Answer: X". Stock token-mean CE over that continuation.
#   View 2 (direct forced choice): the SAME chat prompt + FINAL_CHANNEL_PREFILL, with NO
#     analysis text in context, scored with the same restricted two-letter A/B NLL the
#     --qh-aux arm uses (forced_choice_leg) — pointed at Q_Y only, with NO Q_H sibling.
#
# H-HYGIENE (see GROUNDED_H_HYGIENE_NOTE): Q_H / H_true / H_false / h_target never enter
# the teacher context and never enter ANY supervised field. H text is read ONLY by a
# reject-only audit predicate that can remove or flag an item, never add content.
#
# Rationale, answer-only, and grounded modes use separate artifacts, checkpoint paths,
# and loss implementations.
GROUNDED_PROMPT_VERSION = "grounded-adjudication-v1"

GROUNDED_DEBATERS = ("Debater A", "Debater B")
GROUNDED_CONCLUSION_SUFFIX = " presents the stronger case."

# The TWO schemas are deliberately DIFFERENT so neither can be mistaken for the other:
# parse_grounded_output() accepts only the raw one, assert_rendered_analysis_shape() only
# the rendered one, and each rejects the other (asserted in tests).
GROUNDED_TEACHER_OUTPUT_SCHEMA = (
    'Debater A evidence: "<verbatim span from Debater A\'s arguments>"\n'
    'Debater B evidence: "<verbatim span from Debater B\'s arguments>"\n'
    "Check: <2-4 sentences naming both debaters>\n"
    "Conclusion: <Debater A|Debater B> presents the stronger case."
)
GROUNDED_ANALYSIS_SCHEMA = (
    'Debater A\'s key evidence: "<span>" (<verified|unverified>)\n'
    'Debater B\'s key evidence: "<span>" (<verified|unverified>)\n'
    "Check: <2-4 sentences naming both debaters>\n"
    "Conclusion: <Debater A|Debater B> presents the stronger case."
)

# Audit thresholds are recorded in each artifact and manifest. Loading requires an
# exact match so outputs produced under another threshold set are rejected.
GROUNDED_SPAN_MIN_NORM_CHARS = 25   # measured: median real <quote> is 30 normalized chars
GROUNDED_SPAN_MAX_NORM_CHARS = 300
GROUNDED_CHECK_MIN_CHARS = 120
GROUNDED_CHECK_MAX_CHARS = 600
GROUNDED_CHECK_MIN_SENTENCES = 2
GROUNDED_CHECK_MAX_SENTENCES = 4
GROUNDED_MIN_CHARS = 200
GROUNDED_MAX_CHARS = 900
# Fail-loud bound on the supervised span (900 chars ~ 225 tokens + harmony scaffold).
GROUNDED_MAX_CONT_TOKENS = 400
GROUNDED_BLIND_ATTEMPTS = 2   # attempt 0 greedy, 1 sampled: teacher sees NO gold answer
GROUNDED_GOLD_ATTEMPTS = 2    # attempts 2/3: gold NOTE (debater + answer text, never a letter)
GROUNDED_MAX_ATTEMPTS = GROUNDED_BLIND_ATTEMPTS + GROUNDED_GOLD_ATTEMPTS
GROUNDED_GOLD_TIER_WARN_FRACTION = 0.40
# Worker-pool timing. The parent never blocks forever on a result: it polls, and on every
# timeout re-checks child liveness, so a dead child ends the wait instead of deadlocking.
GROUNDED_RESULT_POLL_SECONDS = 2.0
GROUNDED_CHILD_READY_TIMEOUT = 1800.0   # a 44 GiB bf16 replica can take minutes to load
GROUNDED_CHILD_JOIN_TIMEOUT = 30.0

GROUNDED_UNRESOLVED_FAIL = "fail"
GROUNDED_UNRESOLVED_ANSWER_ONLY = "answer-only"
GROUNDED_UNRESOLVED_POLICIES = (GROUNDED_UNRESOLVED_FAIL, GROUNDED_UNRESOLVED_ANSWER_ONLY)
GROUNDED_UNRESOLVED_DEFAULT = GROUNDED_UNRESOLVED_FAIL
GROUNDED_LAMBDA_LM_DEFAULT = 1.0
GROUNDED_LAMBDA_FC_DEFAULT = 2.0
# Teacher replicas for ``--generate-grounded``. One uses a single local replica; N>1
# loads one replica per GPU and runs N per-item tier ladders concurrently.
GROUNDED_WORKERS_DEFAULT = 1

# ---- grounded API transport (--teacher-backend api) ------------------------------------
# Bounded in-flight HTTP requests for --generate-grounded, mirroring debate-bok.py's
# DEFAULT_CONCURRENCY. Applies to NOTHING else: the transformers teacher is one in-process
# replica and stays single-threaded (grounded_concurrency_of forces 1 there).
GROUNDED_CONCURRENCY_DEFAULT = 16
# Per-request cap for the grounded API teacher ONLY. TEACHER_TIMEOUT (120s) is sized for a
# single unqueued request; with 16 in flight a 4096-token completion routinely queues past
# it, and a timeout burns a ladder attempt. The rationale arm keeps TEACHER_TIMEOUT.
GROUNDED_API_TIMEOUT = 600
# TRANSPORT retries INSIDE one ladder attempt (connection reset, 429, 5xx). Deliberately
# explicit: the OpenAI SDK is constructed with max_retries=0 so no retry is hidden from the
# artifact's attempt accounting (a transport retry never appends to `failures`, never
# advances attempt/tier/temperature/seed, and never adds a telemetry `call`).
GROUNDED_API_TRANSPORT_RETRIES = 4
GROUNDED_API_RETRY_BACKOFF = 2.0
GROUNDED_ENDPOINT_PROBE_TIMEOUT = 10
# Exception TYPE NAMES worth a transport retry, matched by name because `openai` is a lazy
# import and its exception classes must not be imported at module scope (same policy as
# debate-bok.py's _RETRYABLE_EXC_NAMES).
GROUNDED_RETRYABLE_EXC_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "RateLimitError", "InternalServerError", "ConnectionError", "Timeout",
    "ReadTimeout", "ConnectTimeout", "RemoteProtocolError",
})

# Static, arm-owned term lists (copies of the rationale lists so the two arms are
# decoupled and the grounded list lives under GROUNDED_PROMPT_VERSION). A test asserts
# this stays a SUPERSET of the rationale list, so it can never become weaker.
GROUNDED_FORBIDDEN_TERMS_HARD = RATIONALE_FORBIDDEN_TERMS_HARD
GROUNDED_FORBIDDEN_TERMS_IF_ABSENT_FROM_PROMPT = RATIONALE_FORBIDDEN_TERMS_IF_ABSENT_FROM_PROMPT

# RAW teacher schema (parse only).
_GROUNDED_EVIDENCE_LINE_RE = re.compile(r'^Debater (A|B) evidence: "(.+)"$')
_GROUNDED_CHECK_LINE_RE = re.compile(r"^Check: (.+)$")
_GROUNDED_CONCLUSION_LINE_RE = re.compile(
    r"^Conclusion: (Debater A|Debater B) presents the stronger case\.?$"
)
# RENDERED target schema (shape assertion only).
_GROUNDED_RENDERED_EVIDENCE_RE = re.compile(
    r'^Debater (A|B)\'s key evidence: "(.+)" \((verified|unverified)\)$'
)
_GROUNDED_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Letter hygiene. The letter class is UPPERCASE-only on purpose: "answer a reader expects"
# must not trip, "answer A" / "Option B" / "(A)" must.
GROUNDED_LETTER_BAN_RES = (
    re.compile(r"(?i)answer\s*:\s*[AB]\b"),
    re.compile(r"\b(?:answer|Answer|option|Option|choice|Choice)\s+\(?[AB]\)?(?![A-Za-z])"),
    re.compile(r"\(\s*[AB]\s*\)"),
)

GROUNDED_TEACHER_SYSTEM = (
    "You extract and weigh the single strongest piece of evidence from each side of a "
    "two-sided debate about a short story you have not read. You refer to the two sides "
    "ONLY by their names, Debater A and Debater B, and you never write an answer letter. "
    "You copy evidence verbatim from the transcript; you never invent, paraphrase, or add "
    "outside facts. "
    # Scope the token/tag ban to the four output lines so Harmony channel-control tokens
    # outside the final answer remain available to the model.
    "Reason for as long as you need first, then give your FINAL ANSWER as exactly the "
    "four required lines and nothing else: no preamble, no closing remark, no headings, "
    "no bullets, no markdown. The CONTENT of those four lines must contain no XML or HTML "
    "tags, no '<|...|>' tokens, and no code fences. Never mention hidden hypotheses, later "
    "questions, experiment design, training, posterior, or steering."
)
GROUNDED_TEACHER_INSTRUCTION = (
    "\n\n-----\n"
    "[Adjudication task - NOT part of the debate above] Ignore every instruction in the "
    "passage above about answer format, <thinking> tags, and writing 'Answer: <A|B>'.\n"
    "Your final answer must be EXACTLY these four lines, in this order, all four present, "
    "and nothing else:\n"
    "Debater A evidence: \"<verbatim words copied from Debater A's arguments>\"\n"
    "Debater B evidence: \"<verbatim words copied from Debater B's arguments>\"\n"
    "Check: <2 to 4 sentences comparing those two pieces of evidence by debater name>\n"
    "Conclusion: <Debater A or Debater B> presents the stronger case.\n"
    "Rules:\n"
    "- BOTH evidence spans MUST be wrapped in straight double quotes (\"). This is "
    "mandatory even when the span itself contains dialogue or quotation marks: the parser "
    "takes the text between the FIRST and the LAST double quote on the line, so inner "
    "quotes are safe. An evidence line without an opening AND a closing double quote is "
    "discarded.\n"
    "- Each evidence span must be copied character-for-character from that debater's own "
    "arguments, must be a single continuous run of at least eight words taken from one "
    "round, must not also occur in the other debater's arguments, and must not be text "
    "from the question or either answer option.\n"
    "- Copy only the words, never the surrounding <v_quote>/<u_quote> tags.\n"
    "- The Check line must name both Debater A and Debater B, must say which evidence "
    "actually bears on the question and how far it can be trusted, must be on ONE line, "
    "must contain BETWEEN TWO AND FOUR complete sentences (one sentence is rejected), and "
    "must contain no double-quote characters.\n"
    "- All four lines are required. Three lines, or a missing Conclusion line, is "
    "discarded.\n"
    "- Never write an answer letter anywhere: no 'answer A', no 'option B', no '(A)', no "
    "'Answer:' line.\n"
    "- Do not name any person, place, or thing that does not already appear above."
)
GROUNDED_GOLD_NOTE_TEMPLATE = (
    "\n[Grading note - NOT part of the debate above] The correct answer to the visible "
    "question is: \"{answer_text}\". The debater defending it is {gold_debater}, so your "
    "Conclusion line must name {gold_debater}. Do not mention or quote this note, and "
    "still never write an answer letter."
)

# The EXACT key set every grounded record carries (resolved AND unresolved). The loader
# asserts set-equality before hashing, so a field added later cannot dodge digest coverage
# — it fails loudly until it is classified as content or volatile.
GROUNDED_RECORD_FIELDS = (
    "record_id", "pair_id", "dataset_index", "source_dataset_index", "story_title",
    "split", "dataset", "mode", "grounded_prompt_version", "status",
    "teacher_row_id", "teacher_orientation", "teacher_saw_gold", "teacher_tier",
    "teacher_attempt", "teacher_temperature", "teacher_model", "teacher_backend",
    "teacher_prompt_sha256", "evidence", "check_text", "winner_debater", "analysis",
    "analysis_chars", "audit", "audit_thresholds", "base_row_sha256", "grounded_sha256",
    "grounded_row_sha256", "failures", "created_at", "grounded_content_sha256",
)
# Runtime-only: recorded for auditing, never an input to any digest or comparison.
GROUNDED_VOLATILE_FIELDS = ("created_at", "grounded_content_sha256")
GROUNDED_CONTENT_FIELDS = tuple(
    f for f in GROUNDED_RECORD_FIELDS if f not in GROUNDED_VOLATILE_FIELDS
)

GROUNDED_H_HYGIENE_NOTE = (
    "Q_H, H_true, H_false, h_target and all H-derived selection scores NEVER appear in the "
    "teacher context and NEVER enter any supervised artifact field. Every supervised field "
    "— the two evidence spans, their harness-recomputed verification status, the check "
    "text, the winner, and the rendered analysis G — is a deterministic function of "
    "(judge prompt, transcript, story, Y_true, teacher randomness) alone. H text is used "
    "EXCLUSIVELY by a reject-only audit predicate at artifact build/load time, which can "
    "only remove or flag an item and can never add H-derived content to the training "
    "signal. Organic overlaps (H text genuinely present in the visible transcript) are "
    "recorded for analysis and are not an exclusion criterion."
)
GROUNDED_SEMANTICS_NOTE = (
    "This checkpoint was trained with the grounded-adjudication objective: a mechanically "
    "audited, orientation-invariant evidence/check/winner analysis target PLUS a restricted "
    "A/B forced-choice Q_Y loss at the direct final-channel answer slot. " +
    GROUNDED_H_HYGIENE_NOTE +
    " The trained analysis is INTERVENTION-GROUNDED, not faithful chain-of-thought: the "
    "claims it licenses are behavioural (citations are verbatim-valid; decisions are "
    "sensitive to cited evidence), never introspective. This arm is a distinct point in "
    "verifier-design space and must never be mixed into --qh-aux positive-control "
    "aggregates, nor described as evidence about Q_H."
)
GROUNDED_LOSS_FORMULA = (
    "L = lambda_lm * L_LM + lambda_fc * L_FC  (two forwards, one backward on the sum). "
    "L_LM = model(input_ids, attention_mask, labels).loss, i.e. the stock ForCausalLMLoss "
    "token MEAN over the unmasked grounded continuation (analysis channel + final-channel "
    "'Answer: X' + return); with per_device batch 1 this is exactly a per-example token "
    "mean. L_FC = -(z_target - logsumexp(z_A, z_B)) where z_A/z_B are the fp32-cast "
    "final-position logits of the single-token ' A'/' B' completions for the SAME chat "
    "prompt followed by FINAL_CHANNEL_PREFILL and NO analysis text — the identical "
    "restricted two-letter NLL implementation the --qh-aux arm uses, pointed at Q_Y with "
    "NO Q_H sibling. Trainer.model_accepts_loss_kwargs is forced False, so "
    "num_items_in_batch is never passed and Trainer applies gradient-accumulation "
    "normalization to the weighted SUM rather than rescaling either view alone."
)
GROUNDED_QH_AUX_LOSS_FORMULA = (
    "L = lambda_lm * L_grounded_lm + lambda_fc * L_grounded_qy + "
    "qh_lambda * L_qh (three forwards, one backward on the weighted sum). "
    "The first two terms are exactly the grounded arm: the audited grounded analysis "
    "token-mean CE and the analysis-free direct Q_Y restricted A/B NLL. The third term "
    "is the production-template Q_H restricted A/B NLL from --qh-aux. All forced-choice "
    "legs use the same final-position implementation and Trainer normalization guard."
)
GROUNDED_QH_AUX_SEMANTICS_NOTE = (
    "This is the grounded adjudication arm PLUS the explicit Q_H positive-control / "
    "capacity term. View 1 and grounded View 2 are both Q_Y/Y_true-supporting; the Q_H "
    "term is directly supervised and must not be interpreted as transcript-only emergent "
    "posterior steering. Keep this checkpoint separate from both pure-grounded and "
    "pure answer-only aggregates."
)
GROUNDED_LAMBDA_NOTE = (
    "lambda_lm weights a per-example token mean over the grounded continuation; lambda_fc "
    "weights one undiluted decision NLL. The default lambda_fc=2.0 > lambda_lm=1.0 because "
    "View 1's final letter is conditioned on an analysis that already names the winner and "
    "is therefore nearly free, while View 2 is the only term that trains the UNCONDITIONED "
    "decision and the only one tied to preserved Q_Y accuracy at the deployed readout slot."
)


def grounded_audit_thresholds():
    """Every audit constant, as recorded in the artifact/manifest and re-checked on load."""
    return {
        "span_min_norm_chars": GROUNDED_SPAN_MIN_NORM_CHARS,
        "span_max_norm_chars": GROUNDED_SPAN_MAX_NORM_CHARS,
        "check_min_chars": GROUNDED_CHECK_MIN_CHARS,
        "check_max_chars": GROUNDED_CHECK_MAX_CHARS,
        "check_min_sentences": GROUNDED_CHECK_MIN_SENTENCES,
        "check_max_sentences": GROUNDED_CHECK_MAX_SENTENCES,
        "analysis_min_chars": GROUNDED_MIN_CHARS,
        "analysis_max_chars": GROUNDED_MAX_CHARS,
        "blind_attempts": GROUNDED_BLIND_ATTEMPTS,
        "gold_attempts": GROUNDED_GOLD_ATTEMPTS,
    }


# Bytes/param of SHARDED training state with fp32 masters: fp32 params (4) +
# fp32 grads (4) + fp32 Adam moments (8). bf16 gathers/activations counted as
# a flat per-GPU overhead below.
STATE_BYTES_PER_PARAM = 16
PER_GPU_OVERHEAD_GB = 14.0
GPU_MEM_SAFETY = 0.92
MIN_CPU_RAM_GB = 120.0  # rank0 fp32 load (~84 GB) + FULL_STATE_DICT gather headroom

# DeepSpeed ZeRO-3 + optimizer CPU offload path. GPU state is mostly bf16
# sharded params + sharded grads; fp32 optimizer/master state moves to CPU.
# The overhead is intentionally conservative because MoE all-gathers, activation
# checkpointing, allocator fragmentation, and save-time buffers are model/version
# sensitive.
DS_GPU_BYTES_PER_PARAM = 4
DS_PER_GPU_OVERHEAD_GB = 28.0
DS_CPU_OFFLOAD_BYTES_PER_PARAM = 12
DS_MIN_CPU_RAM_GB = 384.0


# ---------------------------------------------------------------------------
# Dataset / artifact modes
# ---------------------------------------------------------------------------

MODE_CONFIG = {
    "honest": {
        "transcript_field": "transcript",
        "artifact_suffix": "",
        "split_name_prefix": "fullft",
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "answer_only_output_dir": DEFAULT_ANSWER_ONLY_OUTPUT_DIR,
        "grounded_output_dir": DEFAULT_GROUNDED_OUTPUT_DIR,
    },
    "adversarial": {
        "transcript_field": "adversarial_transcript",
        "artifact_suffix": "-adversarial",
        "split_name_prefix": "fullft-adversarial",
        "default_output_dir": DEFAULT_ADVERSARIAL_OUTPUT_DIR,
        "answer_only_output_dir": DEFAULT_ADVERSARIAL_ANSWER_ONLY_OUTPUT_DIR,
        "grounded_output_dir": DEFAULT_ADVERSARIAL_GROUNDED_OUTPUT_DIR,
    },
}


def supervision_of(args_or_mode):
    """Return the selected supervision objective; string input selects rationale."""
    if isinstance(args_or_mode, str):
        return SUPERVISION_RATIONALE
    return getattr(args_or_mode, "supervision", SUPERVISION_RATIONALE)


def qh_aux_of(args_or_mode):
    """Return whether the Q_H auxiliary view is enabled."""
    if isinstance(args_or_mode, str):
        return False
    return bool(getattr(args_or_mode, "qh_aux", False))


def qh_lambda_of(args_or_mode):
    """Resolved Q_H auxiliary weight (0.0 when the arm is off)."""
    if not qh_aux_of(args_or_mode):
        return 0.0
    value = getattr(args_or_mode, "qh_lambda", None)
    return QH_AUX_DEFAULT_LAMBDA if value is None else float(value)


def qy_loss_of(args_or_mode):
    """Return the effective Q_Y loss implementation.

    Runs without ``--qh-aux`` use token CE. The paired Q_Y/Q_H auxiliary mode defaults
    to the symmetric restricted-A/B forced-choice loss.
    """
    if isinstance(args_or_mode, str) or not qh_aux_of(args_or_mode):
        return QY_LOSS_TOKEN_CE
    value = getattr(args_or_mode, "qy_loss", None)
    return QY_LOSS_DEFAULT if value is None else value


def grounded_of(args_or_mode):
    """Return whether grounded adjudication is the selected supervision mode."""
    return supervision_of(args_or_mode) == SUPERVISION_GROUNDED


def lambda_lm_of(args_or_mode):
    """Resolved View-1 (grounded LM) weight, or None when the arm is off.

    None-vs-value is how "the user supplied this flag" is detected, exactly like
    --qh-lambda / --qy-loss: argparse stores None by default and this resolver applies
    the documented default only for a grounded run.
    """
    if not grounded_of(args_or_mode):
        return None
    value = getattr(args_or_mode, "lambda_lm", None)
    return GROUNDED_LAMBDA_LM_DEFAULT if value is None else float(value)


def lambda_fc_of(args_or_mode):
    """Resolved View-2 (direct forced-choice) weight, or None when the arm is off."""
    if not grounded_of(args_or_mode):
        return None
    value = getattr(args_or_mode, "lambda_fc", None)
    return GROUNDED_LAMBDA_FC_DEFAULT if value is None else float(value)


def grounded_unresolved_of(args_or_mode):
    """Resolved policy for items the teacher never resolved, or None when off."""
    if not grounded_of(args_or_mode):
        return None
    value = getattr(args_or_mode, "grounded_unresolved", None)
    return GROUNDED_UNRESOLVED_DEFAULT if value is None else value


def grounded_workers_of(args_or_mode):
    """Return the number of local teacher replicas for grounded generation."""
    value = getattr(args_or_mode, "generate_grounded_workers", None)
    return GROUNDED_WORKERS_DEFAULT if value is None else int(value)


def grounded_concurrency_of(args_or_mode):
    """Bounded in-flight HTTP requests for --generate-grounded on the API backend.

    Defaults to GROUNDED_CONCURRENCY_DEFAULT on `api` and to 1 on every other backend, and
    is FORCED to 1 off the api backend whatever the flag says. main() already rejects that
    combination loudly; this is defence in depth, so no future caller can drive a single
    in-process transformers replica (and its process-global torch RNG) from N threads.
    """
    backend = getattr(args_or_mode, "teacher_backend", TEACHER_BACKEND_DEFAULT)
    if backend != "api":
        return 1
    value = getattr(args_or_mode, "generate_grounded_concurrency", None)
    return GROUNDED_CONCURRENCY_DEFAULT if value is None else int(value)


def mode_config(mode_or_args):
    mode = mode_or_args if isinstance(mode_or_args, str) else getattr(mode_or_args, "mode", "honest")
    dataset = (DEFAULT_DATASET if isinstance(mode_or_args, str)
               else getattr(mode_or_args, "dataset", DEFAULT_DATASET))
    supervision = supervision_of(mode_or_args)
    qh_aux = qh_aux_of(mode_or_args)
    qy_loss = qy_loss_of(mode_or_args)
    grounded = grounded_of(mode_or_args)
    unresolved_policy = grounded_unresolved_of(mode_or_args)
    if unresolved_policy is not None and unresolved_policy not in GROUNDED_UNRESOLVED_POLICIES:
        raise ValueError(f"unknown grounded unresolved policy {unresolved_policy!r}")
    try:
        mode_values = MODE_CONFIG[mode]
    except KeyError as exc:
        raise ValueError(f"unknown verifier data mode {mode!r}") from exc
    if supervision not in SUPERVISION_MODES:
        raise ValueError(f"unknown supervision objective {supervision!r}")
    if qy_loss not in QY_LOSS_MODES:
        raise ValueError(f"unknown Q_Y loss {qy_loss!r}")
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset):
        raise ValueError(
            f"invalid dataset name {dataset!r}; --dataset must be a directory name "
            "such as QuALITY-H or GPQA"
        )

    dataset_dir = os.path.join(DATASET_ROOT, dataset)
    dataset_stem = os.path.join(dataset_dir, dataset)
    artifact_stem = f"{dataset_stem}-verifier-fullft{mode_values['artifact_suffix']}"
    if supervision == SUPERVISION_GROUNDED:
        output_template = mode_values["grounded_output_dir"]
    elif supervision == SUPERVISION_ANSWER_ONLY:
        output_template = mode_values["answer_only_output_dir"]
    else:
        output_template = mode_values["default_output_dir"]
    output_dir = output_template.format(dataset=dataset)
    if qh_aux:
        # The Q_H-supervised positive control NEVER shares a default checkpoint dir
        # with the pure arm (guard_checkpoint_identity is the second line of defence).
        output_dir = output_dir.rstrip("/") + QH_AUX_OUTPUT_DIR_SUFFIX

    return {
        **mode_values,
        "dataset": dataset,
        "supervision": supervision,
        "qh_aux": qh_aux,
        "qy_loss": qy_loss,
        "grounded": grounded,
        "grounded_prompt_version": GROUNDED_PROMPT_VERSION if grounded else None,
        "grounded_lambda_lm": lambda_lm_of(mode_or_args),
        "grounded_lambda_fc": lambda_fc_of(mode_or_args),
        "grounded_unresolved_policy": unresolved_policy,
        "grounded_train_jsonl": f"{artifact_stem}-train-grounded.jsonl",
        "grounded_eval_jsonl": f"{artifact_stem}-eval-grounded.jsonl",
        "grounded_manifest_template": f"{artifact_stem}-{{split}}-grounded-manifest.json",
        "qh_target_semantic": QH_AUX_TARGET_SEMANTIC[mode],
        "qh_train_jsonl": f"{artifact_stem}-qh-train.jsonl",
        "qh_eval_jsonl": f"{artifact_stem}-qh-eval.jsonl",
        "dataset_dir": dataset_dir,
        "dataset_path": (
            f"{dataset_stem}-with-honest-transcripts.json"
            if mode == "honest" else f"{dataset_stem}.json"
        ),
        "reference_path": f"{dataset_stem}.json",
        "stories_path": f"{dataset_stem}-title-story.json",
        "split_path": f"{artifact_stem}-split.json",
        "train_jsonl": f"{artifact_stem}-train.jsonl",
        "eval_jsonl": f"{artifact_stem}-eval.jsonl",
        "rationale_train_jsonl": f"{artifact_stem}-train-rationale.jsonl",
        "rationale_eval_jsonl": f"{artifact_stem}-eval-rationale.jsonl",
        "rationale_manifest_template": f"{artifact_stem}-{{split}}-rationale-manifest.json",
        "output_dir": output_dir,
        # The usable-item count is derived per run from the dataset file (see
        # count_transcript_bearing_rows); there is deliberately no constant here.
        "expected_items_policy": (
            "derived from the dataset file: rows whose Q_Y carries a dict "
            f"{mode_values['transcript_field']!r}"
        ),
    }


def _read_json(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} not found: {path!r}. Expected dataset layout: "
            "dataset/<NAME>/<NAME>.json, <NAME>-with-honest-transcripts.json, "
            "and <NAME>-title-story.json."
        )
    with open(path) as f:
        return json.load(f)


def _stable_pair_id(dataset, item):
    """Build an order-independent pair id when the compact dataset has none."""
    dataset_slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")
    identity = json.dumps(
        [
            dataset,
            item["story_title"],
            item["Q_Y"]["question"],
            item["Q_H"]["question"],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_slug}_{digest}"


def load_and_validate(dataset_path, reference_path, dataset):
    """Load the compact paired dataset and attach stable local metadata.

    ``<NAME>.json`` supplies the selected transcript condition and
    ``<NAME>-with-honest-transcripts.json`` supplies the honest reference. Questions and
    answers are cross-checked by story title plus the Q_Y/Q_H question pair. ``pair_id``
    is derived when absent, and ``source_partition`` remains null when unavailable.
    """
    items = _read_json(dataset_path, "dataset file")
    reference = _read_json(reference_path, "reference dataset file")
    if not isinstance(items, list) or not isinstance(reference, list):
        raise ValueError(f"{dataset_path!r} and {reference_path!r} must both contain JSON arrays")

    def index_by_item_key(rows, path):
        indexed = {}
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: item {idx} is not an object")
            title = row.get("story_title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"{path}: item {idx} has empty/missing story_title")
            q_y = row.get("Q_Y")
            q_h = row.get("Q_H")
            if not isinstance(q_y, dict) or not isinstance(q_h, dict):
                raise ValueError(f"{path}: item {idx} has missing/non-object Q_Y or Q_H")
            q_y_question = q_y.get("question")
            q_h_question = q_h.get("question")
            if not isinstance(q_y_question, str) or not q_y_question.strip():
                raise ValueError(f"{path}: item {idx} has empty/missing Q_Y.question")
            if not isinstance(q_h_question, str) or not q_h_question.strip():
                raise ValueError(f"{path}: item {idx} has empty/missing Q_H.question")
            key = (title, q_y_question, q_h_question)
            if key in indexed:
                raise ValueError(f"{path}: duplicate story/question pair at item {idx}")
            indexed[key] = row
        return indexed

    reference_by_key = index_by_item_key(reference, reference_path)
    item_by_key = index_by_item_key(items, dataset_path)
    if set(item_by_key) != set(reference_by_key):
        missing = sorted(set(reference_by_key) - set(item_by_key))
        extra = sorted(set(item_by_key) - set(reference_by_key))
        raise ValueError(
            f"{dataset_path}: story/question-pair coverage differs from {reference_path}; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    field_names = {
        "Q_Y": ("question", "Y_true", "Y_false"),
        "Q_H": ("question", "H_true", "H_false"),
    }
    pair_ids = set()
    for idx, item in enumerate(items):
        title = item["story_title"]
        item_key = (title, item["Q_Y"]["question"], item["Q_H"]["question"])
        reference_item = reference_by_key[item_key]
        where = f"{dataset_path}: item {idx} ({title!r})"
        for q_key, fields in field_names.items():
            holder = item.get(q_key)
            reference_holder = reference_item.get(q_key)
            if not isinstance(holder, dict) or not isinstance(reference_holder, dict):
                raise ValueError(f"{where}: missing/non-object {q_key}")
            for field in fields:
                value = holder.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{where}: empty/missing {q_key}.{field}")
                if value != reference_holder.get(field):
                    raise ValueError(
                        f"{where}: {q_key}.{field} mismatch vs {reference_path}"
                    )

        pair_id = item.get("pair_id")
        if pair_id is None:
            pair_id = reference_item.get("pair_id")
        if pair_id is None:
            pair_id = _stable_pair_id(dataset, item)
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"{where}: pair_id must be a non-empty string when present")
        if pair_id in pair_ids:
            raise ValueError(f"{where}: duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)

        source_partition = item.get("source_partition")
        if source_partition is None:
            source_partition = reference_item.get("source_partition")
        if source_partition is not None and not isinstance(source_partition, str):
            raise ValueError(f"{where}: source_partition must be a string or null")
        item["pair_id"] = pair_id
        item["source_partition"] = source_partition
        item["dataset_index"] = idx

    return items


def load_stories_for_items(items, cfg):
    stories = _read_json(cfg["stories_path"], "title-to-story map")
    if not isinstance(stories, dict):
        raise ValueError(f"{cfg['stories_path']!r} must contain a JSON object")
    bad = [
        title for title, story in stories.items()
        if not isinstance(title, str) or not isinstance(story, str) or not story.strip()
    ]
    if bad:
        raise ValueError(
            f"{cfg['stories_path']}: non-string/empty story values for {bad[:3]}"
        )
    missing = [item["story_title"] for item in items if item["story_title"] not in stories]
    if missing:
        raise ValueError(f"stories missing from {cfg['stories_path']}: {missing[:3]}")
    return stories


def count_transcript_bearing_rows(rows, transcript_field):
    """Count rows whose Q_Y carries a usable transcript under ``transcript_field``.

    ``load_items_for_mode`` uses this input-derived count to verify that its selection
    loop preserves every eligible row without relying on per-dataset constants.
    """
    return sum(
        1 for row in rows
        if isinstance(row, dict) and isinstance(row.get("Q_Y", {}).get(transcript_field), dict)
    )


def load_items_for_mode(mode_or_args):
    """Load Q_Y items for the selected transcript mode and normalize the chosen
    transcript to Q_Y['transcript'] so the existing row/prompt code stays shared.

    In adversarial mode the source dataset may contain null transcripts; only rows
    with a selected adversarial transcript are usable for verifier fine-tuning.
    Dataset indices are reset to the selected-mode list so rationale joins remain
    list-local; the original merged-dataset index is kept as source_dataset_index.
    """
    cfg = mode_config(mode_or_args)
    mode = mode_or_args if isinstance(mode_or_args, str) else getattr(mode_or_args, "mode", "honest")
    items = load_and_validate(cfg["dataset_path"], cfg["reference_path"], cfg["dataset"])
    out = []
    skipped = 0
    for item in items:
        transcript = item["Q_Y"].get(cfg["transcript_field"])
        if transcript is None:
            if mode == "adversarial":
                skipped += 1
                continue
            raise ValueError(
                f"{cfg['dataset_path']}: {item['story_title']!r} missing "
                f"Q_Y.{cfg['transcript_field']}"
            )
        if not isinstance(transcript, dict):
            raise ValueError(
                f"{cfg['dataset_path']}: {item['story_title']!r} has non-dict "
                f"Q_Y.{cfg['transcript_field']}"
            )

        normalized = copy.deepcopy(item)
        if mode == "adversarial":
            normalized["source_dataset_index"] = normalized["dataset_index"]
            normalized["dataset_index"] = len(out)
        normalized["Q_Y"]["transcript"] = copy.deepcopy(transcript)
        out.append(normalized)

    # Integrity check with a DERIVED expectation (never a hardcoded constant): an
    # independent scan of the same dataset file must agree with what the selection
    # loop above kept, and selected + skipped must account for every row.
    source_rows = _read_json(cfg["dataset_path"], "dataset file")
    expected = count_transcript_bearing_rows(source_rows, cfg["transcript_field"])
    if len(out) != expected:
        raise ValueError(
            f"{mode} mode selected {len(out)} usable items but {cfg['dataset_path']} "
            f"contains {expected} rows with a dict Q_Y.{cfg['transcript_field']} "
            f"({skipped} skipped) — the selection loop and the dataset disagree."
        )
    if len(out) + skipped != len(source_rows):
        raise ValueError(
            f"{mode} mode accounted for {len(out)} selected + {skipped} skipped != "
            f"{len(source_rows)} rows in {cfg['dataset_path']}"
        )
    if not out:
        raise ValueError(f"{mode} mode produced no usable transcript items")
    if mode == "adversarial":
        print(f"Mode adversarial: selected {len(out)} items from {cfg['dataset_path']} "
              f"({skipped} skipped with null adversarial_transcript); usable count derived "
              f"from the dataset, not a hardcoded constant.")
    return out


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def transcript_text_of(item):
    """All raw argument text of the item's honest transcript (both debaters,
    all rounds) — used only for the organic H-overlap audit."""
    rounds = item["Q_Y"]["transcript"]["rounds"]
    return "\n".join(rnd[name] for rnd in rounds for name in ("Debater A", "Debater B"))


def h_overlap_flags(item):
    """Audit flags: does the honest Q_Y transcript organically contain the
    H_true / H_false answer text? Case-insensitive substring with a minimum
    length so trivial options ("Yes") don't flag everything. The matched text
    is stored alongside the boolean so the artifact supports inspection without
    recomputation. NOT an exclusion criterion — see module docstring."""
    transcript_lower = transcript_text_of(item).lower()
    flags = {}
    for field in ("H_true", "H_false"):
        text = item["Q_H"][field].strip()
        hit = len(text) >= H_OVERLAP_MIN_CHARS and text.lower() in transcript_lower
        flags[f"transcript_contains_{field.lower()}_text"] = hit
        flags[f"{field.lower()}_text_match"] = text if hit else None
    return flags


def make_split(items):
    """Deterministic 8:2 split over unique stories, keeping shared-story items together."""
    titles = [it["story_title"] for it in items]
    story_titles = list(dict.fromkeys(titles))
    shuffled_story_titles = story_titles.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_story_titles)
    n_train_stories = int(len(story_titles) * TRAIN_FRACTION)
    train_story_titles = set(shuffled_story_titles[:n_train_stories])
    eval_story_titles = set(shuffled_story_titles[n_train_stories:])
    train_indices = [idx for idx, title in enumerate(titles) if title in train_story_titles]
    eval_indices = [idx for idx, title in enumerate(titles) if title in eval_story_titles]

    assert not set(train_indices) & set(eval_indices), "train/eval splits overlap"
    assert sorted(train_indices + eval_indices) == list(range(len(items))), "split is not a partition"
    assert not train_story_titles & eval_story_titles, "train/eval story titles overlap"

    index_info = {
        str(it["dataset_index"]): {
            "pair_id": it["pair_id"],
            "story_title": it["story_title"],
            "source_partition": it.get("source_partition"),
            **({"source_dataset_index": it["source_dataset_index"]}
               if "source_dataset_index" in it else {}),
            **h_overlap_flags(it),
        }
        for it in items
    }
    return {
        "split_seed": SPLIT_SEED,
        "split_policy": (
            f"first-occurrence unique story titles ({len(story_titles)}) are shuffled with "
            f"random.Random(split_seed); first {n_train_stories} stories = train, "
            f"remaining {len(story_titles) - n_train_stories} stories = eval; "
            "all items sharing a story_title stay in the same split"
        ),
        "train_story_count": len(train_story_titles),
        "eval_story_count": len(eval_story_titles),
        "train_story_titles": sorted(train_story_titles),
        "eval_story_titles": sorted(eval_story_titles),
        "train_count": len(train_indices),
        "eval_count": len(eval_indices),
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "index_info": index_info,
    }


# ---------------------------------------------------------------------------
# Row construction (paired orientations + name mapping)
# ---------------------------------------------------------------------------

def map_item_orientation(transcript, answer_a_label):
    """The four-way truth table from the module docstring, as one pure function.

    answer_a_label is the semantic label ("Y_true"/"Y_false") shown as answer A.
    Returns (name_a, name_b, target_letter): name_a is the debater whose stance
    equals answer_a_label, and the target letter is wherever Y_true sits."""
    stance_a = transcript["Debater A"]
    stance_b = transcript["Debater B"]
    if {stance_a, stance_b} != {"Y_true", "Y_false"}:
        raise ValueError(f"non-complementary debater stances: A={stance_a!r} B={stance_b!r}")
    name_a = "Debater A" if stance_a == answer_a_label else "Debater B"
    name_b = "Debater B" if name_a == "Debater A" else "Debater A"
    target_letter = "A" if answer_a_label == "Y_true" else "B"
    return name_a, name_b, target_letter


def row_hash(messages, target):
    """Canonical content hash consumed by the training-time drift guard."""
    canonical = json.dumps([messages, target], sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rationale_text_hash(rationale):
    """sha256 of the rationale text (utf-8)."""
    return hashlib.sha256(rationale.encode("utf-8")).hexdigest()


def rationale_row_hash(messages, target, rationale):
    """Combined drift hash binding the rationale to its exact prompt+target. This is
    the rationale-artifact's tamper guard (the base row_hash covers only the prompt+
    target the rationale was written against)."""
    canonical = json.dumps([messages, target, rationale], sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def teacher_prompt_hash(teacher_messages):
    """sha256 of the canonical teacher prompt (system+user), for reproducibility."""
    canonical = json.dumps(teacher_messages, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def combined_rationale_hash(rationale_rows):
    """Manifest-level hash over the ordered per-row combined hashes; detects a
    partial / reordered / swapped rationale file."""
    joined = "\n".join(r["rationale_row_sha256"] for r in rationale_rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_rows(items, stories_by_title, indices, split_name):
    """Two rows per item (both orientations, fixed render order => byte-identical
    re-runs). The story is fetched only for verify_quotes inside the prompt
    builder."""
    rows = []
    for dataset_index in indices:
        item = items[dataset_index]
        q_y = item["Q_Y"]
        transcript = q_y["transcript"]
        story = stories_by_title[item["story_title"]]
        overlap = h_overlap_flags(item)
        for orientation in ORIENTATIONS:
            answer_a_label = "Y_true" if orientation == "A=Y_true" else "Y_false"
            answer_b_label = "Y_false" if answer_a_label == "Y_true" else "Y_true"
            name_a, name_b, target_letter = map_item_orientation(transcript, answer_a_label)
            prompt = build_judge_user_content_mapped(
                q_y["question"],
                q_y[answer_a_label],
                q_y[answer_b_label],
                name_a,
                name_b,
                transcript["rounds"],
                story,
            )
            target = TARGET_TEMPLATE.format(letter=target_letter)
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
            rows.append({
                "row_id": f"{item['pair_id']}|{orientation}",
                "dataset_index": dataset_index,
                "pair_id": item["pair_id"],
                "story_title": item["story_title"],
                "split": split_name,
                "q_key": "Q_Y",
                "orientation": orientation,
                "answer_order": {"A": answer_a_label, "B": answer_b_label},
                "debater_stances": {"Debater A": transcript["Debater A"],
                                    "Debater B": transcript["Debater B"]},
                "debater_to_letter": {name_a: "A", name_b: "B"},
                "name_a": name_a,
                "name_b": name_b,
                "target_letter": target_letter,
                "target": target,
                "messages": messages,
                "prompt_chars": len(prompt),
                **overlap,
                "row_sha256": row_hash(messages, target),
            })
            if "source_dataset_index" in item:
                rows[-1]["source_dataset_index"] = item["source_dataset_index"]
    return rows


# ---------------------------------------------------------------------------
# Leak guards + sanity checks (all torch-free)
# ---------------------------------------------------------------------------

def forbidden_content_checks(rows, items, stories_by_title):
    """Hard-fail on CONSTRUCTED leaks; audit-report organic H-text overlap."""
    from judge_common import normalize_text

    literal_tokens = ("h_target", "H_true", "H_false", "Y_true", "Y_false")
    for row in rows:
        prompt = row["messages"][0]["content"]
        where = row["row_id"]
        item = items[row["dataset_index"]]

        if item["Q_H"]["question"].strip() in prompt:
            raise AssertionError(f"leak guard: Q_H question text in prompt ({where})")
        if "hidden hypothesis" in prompt.lower():
            raise AssertionError(f"leak guard: 'hidden hypothesis' in prompt ({where})")
        for token in literal_tokens:
            if token in prompt:
                raise AssertionError(f"leak guard: literal token {token!r} in prompt ({where})")
        story_norm = normalize_text(stories_by_title[item["story_title"]])
        if story_norm and story_norm in prompt:
            raise AssertionError(f"leak guard: full story text in prompt ({where})")

    overlap_items = sorted(
        {(r["dataset_index"], r["pair_id"], r["split"],
          r["h_true_text_match"], r["h_false_text_match"])
         for r in rows
         if r["transcript_contains_h_true_text"] or r["transcript_contains_h_false_text"]}
    )
    print(f"Leak guard passed on {len(rows)} prompts (no Q_H question / label tokens / "
          f"story text / 'hidden hypothesis').")
    print(f"Organic H-text overlap (audited, NOT excluded): {len(overlap_items)} items:")
    for idx, pair_id, split_name, h_t, h_f in overlap_items:
        matches = [f"{k}={v!r}" for k, v in (("H_true", h_t), ("H_false", h_f)) if v]
        print(f"  index {idx:>3}  {pair_id}  [{split_name}]  transcript contains "
              f"{'; '.join(matches)}")
    return overlap_items


def sanity_check_rows(train_rows, eval_rows, split):
    n_train_items = split["train_count"]
    n_eval_items = split["eval_count"]
    assert len(train_rows) == 2 * n_train_items, (len(train_rows), n_train_items)
    assert len(eval_rows) == 2 * n_eval_items, (len(eval_rows), n_eval_items)

    for name, rows, n_items in (("train", train_rows, n_train_items),
                                ("eval", eval_rows, n_eval_items)):
        dist = Counter(r["target_letter"] for r in rows)
        assert dist["A"] == dist["B"] == n_items, f"{name} rows unbalanced: {dict(dist)}"
        bad = {r["target"] for r in rows} - {"Answer: A", "Answer: B"}
        assert not bad, f"{name}: unexpected targets {bad}"
        by_index = {}
        for r in rows:
            by_index.setdefault(r["dataset_index"], set()).add(r["orientation"])
        missing = {i: v for i, v in by_index.items() if v != set(ORIENTATIONS)}
        assert not missing, f"{name}: items missing an orientation: {missing}"
        print(f"{name}: {n_items} items x 2 orientations = {len(rows)} rows; "
              f"targets A={dist['A']} B={dist['B']} (balanced).")

    # Truth-table verification on real items: every (stance, orientation) combo
    # that occurs must carry the expected name mapping AND the rendered prompt
    # must contain the matching "<NAME_A> is arguing for answer A" line.
    expected = {
        ("Y_true", "A=Y_true"): ("Debater A", "Debater B", "A"),
        ("Y_true", "A=Y_false"): ("Debater B", "Debater A", "B"),
        ("Y_false", "A=Y_true"): ("Debater B", "Debater A", "A"),
        ("Y_false", "A=Y_false"): ("Debater A", "Debater B", "B"),
    }
    seen = set()
    for r in train_rows + eval_rows:
        key = (r["debater_stances"]["Debater A"], r["orientation"])
        exp_name_a, exp_name_b, exp_letter = expected[key]
        assert (r["name_a"], r["name_b"], r["target_letter"]) == (exp_name_a, exp_name_b, exp_letter), (
            f"truth-table violation in {r['row_id']}: {key} -> "
            f"({r['name_a']}, {r['name_b']}, {r['target_letter']})"
        )
        stance_line = f"{r['name_a']} is arguing for answer A, and {r['name_b']} is arguing for answer B."
        assert stance_line in r["messages"][0]["content"], (
            f"stance line missing/mismatched in prompt for {r['row_id']}"
        )
        seen.add(key)
    assert seen == set(expected), f"not all truth-table combos occur in data: missing {set(expected) - seen}"
    print(f"Truth table verified on real items: all 4 (stance x orientation) combos occur "
          f"and map correctly; stance line present in every prompt.")

    lengths = sorted(r["prompt_chars"] for r in train_rows + eval_rows)
    print(f"Prompt chars min/median/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")


def print_sample_prompts(rows):
    """One full sample per stance type so the name swap is visible to a human."""
    shown = set()
    for r in rows:
        stance = r["debater_stances"]["Debater A"]
        if stance in shown:
            continue
        shown.add(stance)
        print(f"\n{'=' * 78}\nSAMPLE ROW {r['row_id']}  (Debater A argues {stance}, "
              f"orientation {r['orientation']}, target {r['target']!r})\n{'=' * 78}")
        print(r["messages"][0]["content"])
        print(f"--- TARGET: {r['target']!r}")
        if shown == {"Y_true", "Y_false"}:
            break


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def write_json_atomic(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_jsonl_atomic(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def verify_artifacts_match(rows, jsonl_path):
    """Drift guard: the cluster must train on byte-identical rows to the locally
    audited artifacts. Compares (row_id, row_sha256) sequences."""
    if not os.path.exists(jsonl_path):
        raise RuntimeError(
            f"{jsonl_path} not found. Run `--check-data` first (locally is fine) and "
            "ship the dataset/ artifacts with the repo — training refuses to run on "
            "unaudited data."
        )
    with open(jsonl_path) as f:
        on_disk = [json.loads(line) for line in f]
    built = [(r["row_id"], r["row_sha256"]) for r in rows]
    stored = [(r["row_id"], r["row_sha256"]) for r in on_disk]
    if built != stored:
        diffs = [i for i, (b, s) in enumerate(zip(built, stored)) if b != s]
        raise RuntimeError(
            f"Audit drift: rebuilt rows differ from {jsonl_path} "
            f"(len {len(built)} vs {len(stored)}; first diffs at rows {diffs[:5]}). "
            "The dataset or prompt template changed since --check-data. Re-run "
            "--check-data, re-audit, and retrain on the regenerated artifacts."
        )
    print(f"Drift guard OK: {len(rows)} rebuilt rows match {jsonl_path} exactly.")


def build_all(write_artifacts, args):
    """Shared by --check-data and train(): load, split, render, check."""
    cfg = mode_config(args)
    items = load_items_for_mode(args)
    stories_by_title = load_stories_for_items(items, cfg)

    split = make_split(items)
    split["dataset"] = cfg["dataset"]
    split["dataset_path"] = cfg["dataset_path"]
    split["stories_path"] = cfg["stories_path"]
    print(f"Split (seed {SPLIT_SEED}): {split['train_count']} train / "
          f"{split['eval_count']} eval over {len(items)} {cfg['dataset']} items "
          f"across {split['train_story_count']} train / "
          f"{split['eval_story_count']} eval stories (disjoint).")

    split_prefix = cfg["split_name_prefix"]
    train_rows = build_rows(items, stories_by_title, split["train_indices"], f"{split_prefix}-train")
    eval_rows = build_rows(items, stories_by_title, split["eval_indices"], f"{split_prefix}-eval")

    forbidden_content_checks(train_rows + eval_rows, items, stories_by_title)
    sanity_check_rows(train_rows, eval_rows, split)

    if write_artifacts:
        write_json_atomic(cfg["split_path"], split)
        write_jsonl_atomic(cfg["train_jsonl"], train_rows)
        write_jsonl_atomic(cfg["eval_jsonl"], eval_rows)
        print(f"Wrote {cfg['split_path']}, {cfg['train_jsonl']} ({len(train_rows)} rows), "
              f"{cfg['eval_jsonl']} ({len(eval_rows)} rows).")
    return split, train_rows, eval_rows


# ---------------------------------------------------------------------------
# Q_H auxiliary rows: schedule, prompts, leak guards, artifacts (torch-free)
# ---------------------------------------------------------------------------
# EVERYTHING in this section runs ONLY under --qh-aux. The pure Q_Y path never
# calls it, so Q_Y rows/bytes/hashes and the rationale artifacts are untouched.

_QH_HARMONY_RE = re.compile(r"<\|[^|]*\|>")
_QH_ANSWER_LINE_RE = re.compile(r"(?i)answer:\s*[AB]\b")
# An assistant/system/developer TURN header, not the words themselves: harmony
# turns look like "<|start|>assistant" and chat-ish injections like "assistant:".
_QH_ROLE_TURN_RE = re.compile(r"(?im)^[ \t]*(?:assistant|system|developer)[ \t]*:")
QH_LITERAL_LEAK_TOKENS = ("h_target", "H_true", "H_false", "Y_true", "Y_false")


def qh_target_semantic_for_mode(mode):
    """Mode lock: honest -> H_true, adversarial -> H_false (no v1 override)."""
    try:
        return QH_AUX_TARGET_SEMANTIC[mode]
    except KeyError as exc:
        raise ValueError(f"unknown verifier data mode {mode!r}") from exc


def qh_schedule_bit(pair_id):
    """Deterministic, MODE-INDEPENDENT letter-schedule bit for one item.

    Mode independence is the point: honest and adversarial rows for the same item
    put their (different) target SEMANTIC at the SAME letter positions, so any
    letter/position bias is identical across arms and only the semantics differ.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError(f"pair_id must be a non-empty string, got {pair_id!r}")
    key = f"{SPLIT_SEED}|{pair_id}|{QH_AUX_SCHEDULE_VERSION}"
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2


def qh_letter_schedule(pair_id):
    """Target LETTER for pair slot 0 and slot 1 (ORIENTATIONS order)."""
    return ("A", "B") if qh_schedule_bit(pair_id) == 0 else ("B", "A")


def qh_row_plan(target_semantic, pair_id, slot):
    """The whole (mode x schedule bit x pair slot) truth table as one pure function.

    The scheduled letter carries the mode's target semantic; the other H option is
    placed opposite. Across the two slots an item therefore always shows BOTH
    letters as target exactly once AND both semantic option orders exactly once.
    """
    if target_semantic not in QH_AUX_SEMANTIC_LABELS:
        raise ValueError(f"unknown Q_H target semantic {target_semantic!r}")
    if slot not in (0, 1):
        raise ValueError(f"pair slot must be 0 or 1, got {slot!r}")
    other_semantic = "H_false" if target_semantic == "H_true" else "H_true"
    target_letter = qh_letter_schedule(pair_id)[slot]
    return {
        "schedule_bit": qh_schedule_bit(pair_id),
        "target_letter": target_letter,
        "target_semantic": target_semantic,
        "answer_order": {
            "A": target_semantic if target_letter == "A" else other_semantic,
            "B": target_semantic if target_letter == "B" else other_semantic,
        },
    }


def qh_provenance_hash(row):
    """sha256 over the CANONICALIZED Q_H row, minus the provenance hash field itself.

    Whole-row coverage is deliberate: an allow-list would silently stop covering any
    field added later. The required-field check only guarantees that a row cannot dodge
    coverage by omitting provenance fields altogether.
    """
    missing = [field for field in QH_REQUIRED_PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"Q_H row {row.get('row_id')!r} is missing provenance field(s) {missing}; "
            "refusing to hash an under-specified audit record."
        )
    payload = {k: v for k, v in row.items() if k != QH_ROW_PROVENANCE_HASH_FIELD}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def qh_item_id(item):
    """PRODUCTION item id: always the canonical merged-dataset position.

    load_items_for_mode renumbers dataset_index to the selected-mode list in
    adversarial mode and keeps the canonical position in source_dataset_index, so
    indexing by dataset_index there would silently disagree with the candidate /
    selection and downstream readout artifacts.
    """
    return adv_common.item_id_for(item.get("source_dataset_index", item["dataset_index"]))


def _qh_reference_positions(cfg):
    """{(story_title, Q_Y question, Q_H question): canonical index} from the reference file."""
    reference = _read_json(cfg["reference_path"], "reference dataset file")
    positions = {}
    for idx, row in enumerate(reference):
        key = (row["story_title"], row["Q_Y"]["question"], row["Q_H"]["question"])
        positions[key] = idx
    return positions


def assert_qh_item_id_positions(items, cfg):
    """Fail loudly unless the index used for item_id IS the canonical position.

    Prompt byte-parity CANNOT catch a wrong item_id (both letter orders are always
    generated and the prompt text carries no id), so this positional cross-check is
    the only guard that the production item_id convention actually holds.
    """
    positions = _qh_reference_positions(cfg)
    for item in items:
        key = (item["story_title"], item["Q_Y"]["question"], item["Q_H"]["question"])
        canonical = positions.get(key)
        used = item.get("source_dataset_index", item["dataset_index"])
        if canonical is None or canonical != used:
            raise RuntimeError(
                f"Q_H aux item-id parity failure for {item['story_title']!r}: "
                f"canonical position in {cfg['reference_path']} is {canonical!r} but this run "
                f"would publish item_id {adv_common.item_id_for(used)!r} (index {used}). "
                "The training dataset file is ordered differently from the canonical dataset, "
                "so production item ids cannot be reproduced. Regenerate the dataset files in "
                "canonical order before running --qh-aux."
            )
    return len(items)


def qh_candidate_for_item(item):
    """A PRODUCTION-shaped candidate dict for score_verifier's readout helpers."""
    q_y, q_h = item["Q_Y"], item["Q_H"]
    return {
        "item_id": qh_item_id(item),
        "story_title": item["story_title"],
        "qy": {"question": q_y["question"], "Y_true": q_y["Y_true"], "Y_false": q_y["Y_false"]},
        "qh": {"question": q_h["question"], "H_true": q_h["H_true"], "H_false": q_h["H_false"]},
        "transcript": q_y["transcript"],
    }


def render_qh_prompt(candidate, story, answer_order, rendered_transcript=None):
    """Render the production Q_H readout prompt for an explicit answer order.

    Mirrors score_verifier.build_qh_prompt exactly, except that the A/B order is
    supplied instead of derived, so both orders can be produced. The rendered
    transcript may be passed in to avoid re-running quote verification per row.
    """
    transcript = candidate["transcript"]
    if rendered_transcript is None:
        rendered_transcript = render_verified_transcript(candidate, story)
    return QH_READOUT_TEMPLATE.format(
        visible_question=candidate["qy"]["question"],
        debater_a_answer=candidate["qy"][transcript["Debater A"]],
        debater_b_answer=candidate["qy"][transcript["Debater B"]],
        transcript=rendered_transcript,
        question=candidate["qh"]["question"],
        answer_a=candidate["qh"][answer_order["A"]],
        answer_b=candidate["qh"][answer_order["B"]],
    )


def build_qh_rows(items, stories_by_title, indices, split_name, args):
    """Two Q_H sibling rows per item, paired 1:1 with the two Q_Y rows.

    Row order is exactly build_rows' order (item, then ORIENTATIONS), so the legs
    line up positionally as well as by row id. The target letter is NEVER part of
    the prompt: it is carried alongside and consumed only by the loss.
    """
    cfg = mode_config(args)
    target_semantic = cfg["qh_target_semantic"]
    rows = []
    for dataset_index in indices:
        item = items[dataset_index]
        candidate = qh_candidate_for_item(item)
        story = stories_by_title[item["story_title"]]
        rendered = render_verified_transcript(candidate, story)  # once per item
        production_order = deterministic_answer_order(candidate, "Q_H", QH_OPTION_SEED)
        for slot, orientation in enumerate(ORIENTATIONS):
            plan = qh_row_plan(target_semantic, item["pair_id"], slot)
            answer_order = plan["answer_order"]
            prompt = render_qh_prompt(candidate, story, answer_order, rendered)
            qy_target_letter = "A" if orientation == "A=Y_true" else "B"
            messages = [{"role": "user", "content": prompt}]
            row = {
                "row_id": f"{item['pair_id']}|{orientation}|Q_H",
                "qy_row_id": f"{item['pair_id']}|{orientation}",
                "dataset_index": dataset_index,
                "pair_id": item["pair_id"],
                "item_id": candidate["item_id"],
                "story_title": item["story_title"],
                "split": split_name,
                "q_key": "Q_H",
                "orientation": orientation,
                "pair_slot": slot,
                "qh_schedule_version": QH_AUX_SCHEDULE_VERSION,
                "qh_schedule_bit": plan["schedule_bit"],
                "answer_order": answer_order,
                "target_semantic": target_semantic,
                "target_letter": plan["target_letter"],
                # Provenance only. This string is NEVER appended to the prompt.
                "target_completion": LABEL_COMPLETIONS[plan["target_letter"]],
                "qy_target_letter": qy_target_letter,
                "letter_aligned_with_qy": plan["target_letter"] == qy_target_letter,
                "production_answer_order": production_order,
                "matches_production_order": answer_order == production_order,
                "readout_prompt_version": QH_READOUT_PROMPT_VERSION,
                "messages": messages,
                "prompt_chars": len(prompt),
                "row_sha256": row_hash(messages, plan["target_letter"]),
            }
            if "source_dataset_index" in item:
                row["source_dataset_index"] = item["source_dataset_index"]
            # LAST: covers every field above, including source_dataset_index.
            row[QH_ROW_PROVENANCE_HASH_FIELD] = qh_provenance_hash(row)
            rows.append(row)
    return rows


def qh_forbidden_content_checks(qh_rows, items, stories_by_title):
    """Hard-fail on CONSTRUCTED leaks in the Q_H prompt; audit organic text.

    The Q_H prompt is SUPPOSED to contain the Q_H question and both H option texts
    — that is the readout itself. What must never appear is constructed leakage:
    dataset label tokens, project jargon, the full story, harmony control tokens or
    assistant-role turns inside the user text, or anything appended after the
    template's fixed instruction tail. Organic "Answer: A/B" prose written by the
    debaters is NOT a leak (the letters mean nothing without the option order that
    only this prompt fixes); it is counted and reported, never rejected.
    """
    from judge_common import normalize_text

    organic_answer_rows = []
    for row in qh_rows:
        messages = row["messages"]
        where = row["row_id"]
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise AssertionError(
                f"Q_H row {where} must be exactly one user message (no assistant turn, "
                "no pre-filled target)"
            )
        prompt = messages[0]["content"]
        item = items[row["dataset_index"]]

        control = _QH_HARMONY_RE.findall(prompt)
        if control:
            raise AssertionError(
                f"leak guard: harmony control token(s) {control[:3]} inside the Q_H user "
                f"prompt ({where}); the readout scaffold must be appended at the ID level only"
            )
        if _QH_ROLE_TURN_RE.search(prompt):
            raise AssertionError(f"leak guard: assistant/system role turn in Q_H prompt ({where})")
        if "hidden hypothesis" in prompt.lower():
            raise AssertionError(f"leak guard: 'hidden hypothesis' in Q_H prompt ({where})")
        for token in QH_LITERAL_LEAK_TOKENS:
            if token in prompt:
                raise AssertionError(
                    f"leak guard: literal token {token!r} in Q_H prompt ({where})"
                )
        story_norm = normalize_text(stories_by_title[item["story_title"]])
        if story_norm and story_norm in prompt:
            raise AssertionError(f"leak guard: full story text in Q_H prompt ({where})")

        if item["Q_H"]["question"].strip() not in prompt:
            raise AssertionError(f"Q_H prompt is missing its Q_H question ({where})")
        for field in QH_AUX_SEMANTIC_LABELS:
            if item["Q_H"][field].strip() not in prompt:
                raise AssertionError(f"Q_H prompt is missing its {field} option text ({where})")
        # Structural "nothing appended": the prompt must still end with the
        # template's own instruction tail, so no target letter/semantic, no
        # "Answer:" scaffold and no completion can have been interpolated.
        if not prompt.endswith(QH_TEMPLATE_TAIL):
            raise AssertionError(
                f"Q_H prompt does not end with the readout template's instruction tail "
                f"({where}); something was appended to the user message"
            )
        hits = _QH_ANSWER_LINE_RE.findall(prompt)
        if hits:
            organic_answer_rows.append((where, len(hits)))

    print(f"Q_H leak guard passed on {len(qh_rows)} prompts (no harmony control tokens / role "
          "turns / label tokens / story text / 'hidden hypothesis' / appended target).")
    if organic_answer_rows:
        total = sum(n for _, n in organic_answer_rows)
        print(f"Organic 'Answer: A/B' text in public transcript (audited, NOT a leak): "
              f"{total} occurrence(s) in {len(organic_answer_rows)} rows, e.g. "
              f"{[r for r, _ in organic_answer_rows[:3]]}")
    return organic_answer_rows


def pair_qy_qh_rows(qy_rows, qh_rows):
    """Zip the two legs, asserting a strict 1:1 positional AND row-id pairing."""
    if len(qy_rows) != len(qh_rows):
        raise RuntimeError(
            f"Q_H aux pairing failure: {len(qy_rows)} Q_Y rows vs {len(qh_rows)} Q_H rows"
        )
    seen = set()
    pairs = []
    for qy, qh in zip(qy_rows, qh_rows):
        if qh["qy_row_id"] != qy["row_id"]:
            raise RuntimeError(
                f"Q_H aux pairing failure: Q_H row {qh['row_id']!r} claims Q_Y row "
                f"{qh['qy_row_id']!r} but is positioned against {qy['row_id']!r}"
            )
        if qh["qy_row_id"] in seen:
            raise RuntimeError(f"duplicate Q_H sibling for Q_Y row {qh['qy_row_id']!r}")
        seen.add(qh["qy_row_id"])
        pairs.append((qy, qh))
    return pairs


def assert_qh_prompt_parity(item_rows, item, story):
    """Both rows must re-render exactly, and ONE must byte-match the production
    score_verifier.build_qh_prompt for its deterministic order."""
    candidate = qh_candidate_for_item(item)
    for row in item_rows:
        expected = render_qh_prompt(candidate, story, row["answer_order"])
        if row["messages"][0]["content"] != expected:
            raise AssertionError(
                f"Q_H prompt for {row['row_id']} does not re-render from (item, answer order) "
                "alone — something item-external was interpolated"
            )
    production_prompt, production_order = build_qh_prompt(candidate, story, QH_OPTION_SEED)
    matches = [r for r in item_rows if r["answer_order"] == production_order]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one Q_H row in the production answer order {production_order} "
            f"for item {item['pair_id']!r}, got {len(matches)}"
        )
    if matches[0]["messages"][0]["content"] != production_prompt:
        raise AssertionError(
            f"Q_H prompt for {matches[0]['row_id']} is NOT byte-identical to "
            "adversarial_transcript.score_verifier.build_qh_prompt; the training and "
            "analysis readouts would differ"
        )
    if not matches[0]["matches_production_order"]:
        raise AssertionError(f"{matches[0]['row_id']}: matches_production_order flag is wrong")


def sanity_check_qh_rows(qh_train_rows, qh_eval_rows, qy_train_rows, qy_eval_rows,
                         items, stories_by_title, split):
    """Balance, schedule re-derivation, pairing and production prompt parity."""
    for name, qh_rows, qy_rows, n_items in (
        ("train", qh_train_rows, qy_train_rows, split["train_count"]),
        ("eval", qh_eval_rows, qy_eval_rows, split["eval_count"]),
    ):
        assert len(qh_rows) == 2 * n_items, (len(qh_rows), n_items)
        pair_qy_qh_rows(qy_rows, qh_rows)

        by_index = {}
        for row in qh_rows:
            by_index.setdefault(row["dataset_index"], []).append(row)
        assert len(by_index) == n_items, (len(by_index), n_items)
        for dataset_index, item_rows in by_index.items():
            item_rows = sorted(item_rows, key=lambda r: r["pair_slot"])
            assert [r["pair_slot"] for r in item_rows] == [0, 1], item_rows[0]["row_id"]
            letters = sorted(r["target_letter"] for r in item_rows)
            assert letters == ["A", "B"], f"{item_rows[0]['row_id']}: target letters {letters}"
            orders = {json.dumps(r["answer_order"], sort_keys=True) for r in item_rows}
            assert len(orders) == 2, f"{item_rows[0]['row_id']}: both semantic orders required"
            semantics = {r["target_semantic"] for r in item_rows}
            assert len(semantics) == 1, f"{item_rows[0]['row_id']}: mixed target semantics"
            # Re-derive the schedule from pair_id ONLY: this is the runtime proof that
            # the letter schedule is mode-independent (it never sees the mode).
            expected = qh_letter_schedule(item_rows[0]["pair_id"])
            got = tuple(r["target_letter"] for r in item_rows)
            assert got == expected, (
                f"{item_rows[0]['row_id']}: letter schedule {got} != re-derived {expected}"
            )
            assert_qh_prompt_parity(item_rows, items[dataset_index],
                                    stories_by_title[items[dataset_index]["story_title"]])

        letter_dist = Counter(r["target_letter"] for r in qh_rows)
        assert letter_dist["A"] == letter_dist["B"] == n_items, dict(letter_dist)
        order_dist = Counter(r["answer_order"]["A"] for r in qh_rows)
        assert order_dist["H_true"] == order_dist["H_false"] == n_items, dict(order_dist)
        print(f"Q_H {name}: {n_items} items x 2 rows = {len(qh_rows)}; target letters "
              f"A={letter_dist['A']} B={letter_dist['B']}, H_true shown as answer A in "
              f"{order_dist['H_true']} rows (both balanced); schedule re-derived from pair_id; "
              "production prompt parity verified per item.")

    lengths = sorted(r["prompt_chars"] for r in qh_train_rows + qh_eval_rows)
    if lengths:
        print(f"Q_H prompt chars min/median/max = "
              f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")


def qh_alignment_audit(pairs, dataset=None):
    """AUDIT ONLY (never gates): how often the Q_H target letter coincides with the
    Q_Y target letter, and with the production deterministic order.

    The schedule bit is drawn per ITEM, so both of an item's pairs are either aligned
    or anti-aligned and the realised fraction is a Binomial(n_items, 1/2)/n_items draw
    around 0.5. Deviation outside QH_ALIGNMENT_WARN_BAND is a SOFT WARNING on stderr
    (measured value + context), never a failure: smaller datasets such as GPQA sit
    further from 0.5 purely by finite-sample luck, and because the schedule is
    MODE-INDEPENDENT the same imbalance appears in the honest and adversarial arms and
    cancels in the between-arm contrast.
    """
    total = len(pairs)
    aligned = sum(1 for _, qh in pairs if qh["letter_aligned_with_qy"])
    production = sum(1 for _, qh in pairs if qh["matches_production_order"])
    n_items = len({qh["pair_id"] for _, qh in pairs})
    fraction = (aligned / total) if total else None
    deviation = abs(fraction - QH_ALIGNMENT_TARGET) if total else None
    # 1 sd of a fair per-item coin, for context in the warning.
    sd = (0.5 / math.sqrt(n_items)) if n_items else None
    audit = {
        "pairs": total,
        "items": n_items,
        "letter_aligned_with_qy": aligned,
        "letter_aligned_with_qy_fraction": round(fraction, 6) if total else None,
        "matches_production_order": production,
        "matches_production_order_fraction": round(production / total, 6) if total else None,
        "alignment_target": QH_ALIGNMENT_TARGET,
        "alignment_warn_band": QH_ALIGNMENT_WARN_BAND,
        "alignment_deviation": round(deviation, 6) if total else None,
        "alignment_fair_coin_sd": round(sd, 6) if sd else None,
        "alignment_warning": None,
    }
    if not total:
        return audit

    print(f"Q_H alignment audit (NOT a filter): Q_H target letter equals the paired Q_Y "
          f"target letter in {aligned}/{total} = "
          f"{audit['letter_aligned_with_qy_fraction']:.3f} of pairs over {n_items} items; "
          f"rows in the production deterministic order: {production}/{total} = "
          f"{audit['matches_production_order_fraction']:.3f}.")
    if deviation > QH_ALIGNMENT_WARN_BAND:
        label = f"for {dataset} " if dataset else ""
        warning = (
            f"Q_H/Q_Y letter-alignment rate {label}is {fraction:.3f} "
            f"({aligned}/{total} pairs over {n_items} items), which deviates from "
            f"{QH_ALIGNMENT_TARGET:.2f} by {deviation:.3f} > band "
            f"{QH_ALIGNMENT_WARN_BAND:.2f} (~{deviation / sd:.1f} sd of a fair per-item "
            f"coin, sd={sd:.3f}). This is EXPECTED on smaller datasets such as GPQA: the "
            "schedule bit is a per-item hash draw, so the realised fraction is binomial "
            "noise, not a bug. It is also MODE-INDEPENDENT, so the identical imbalance "
            "appears in the honest and adversarial arms and cancels in the between-arm "
            "contrast. Proceeding (soft warning; this never fails the build)."
        )
        audit["alignment_warning"] = warning
        print(f"WARNING: {warning}", file=sys.stderr)
    return audit


def qh_combined_hash(rows):
    """Ordered chain over the per-row PROVENANCE hashes; detects a partial, reordered
    or field-forged artifact (row_sha256 alone would miss the latter)."""
    joined = "\n".join(r[QH_ROW_PROVENANCE_HASH_FIELD] for r in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def verify_qh_artifacts_match(rows, jsonl_path):
    """Drift guard for the Q_H artifacts: identical rows AND non-stale contracts.

    Two independent checks:
      * self-consistency — each stored row's provenance hash must match a recomputation
        from its own content, so a lazily edited artifact is caught on its own terms;
      * rebuild identity — (row_id, row_sha256, provenance) must equal the freshly
        rebuilt sequence, so a re-hashed modification is rejected.
    """
    if not os.path.exists(jsonl_path):
        raise RuntimeError(
            f"{jsonl_path} not found. The Q_H auxiliary rows are built and audited by "
            "`--check-data --qh-aux` (locally is fine); training refuses to run on unaudited "
            "Q_H data."
        )
    with open(jsonl_path) as f:
        on_disk = [json.loads(line) for line in f if line.strip()]
    for stored_row in on_disk:
        claimed = stored_row.get(QH_ROW_PROVENANCE_HASH_FIELD)
        try:
            recomputed = qh_provenance_hash(stored_row)
        except ValueError as exc:
            raise RuntimeError(f"{jsonl_path}: {exc}") from exc
        if claimed != recomputed:
            raise RuntimeError(
                f"{jsonl_path}: row {stored_row.get('row_id')!r} carries "
                f"{QH_ROW_PROVENANCE_HASH_FIELD}={claimed!r} but its own content hashes to "
                f"{recomputed!r} — the audit record was edited after it was written."
            )

    def _identity(row):
        return (row["row_id"], row["row_sha256"], row[QH_ROW_PROVENANCE_HASH_FIELD])

    built = [_identity(r) for r in rows]
    stored = [_identity(r) for r in on_disk]
    if built != stored:
        diffs = [i for i, (b, s) in enumerate(zip(built, stored)) if b != s]
        raise RuntimeError(
            f"Q_H audit drift: rebuilt rows differ from {jsonl_path} "
            f"(len {len(built)} vs {len(stored)}; first diffs at rows {diffs[:5]}). "
            "Re-run `--check-data --qh-aux`, re-audit, and retrain on the regenerated artifacts."
        )
    expected_semantic = rows[0]["target_semantic"] if rows else None
    for stored_row in on_disk:
        if stored_row.get("qh_schedule_version") != QH_AUX_SCHEDULE_VERSION:
            raise RuntimeError(
                f"{jsonl_path}: qh_schedule_version {stored_row.get('qh_schedule_version')!r} != "
                f"{QH_AUX_SCHEDULE_VERSION!r}; regenerate this stale artifact"
            )
        if stored_row.get("readout_prompt_version") != QH_READOUT_PROMPT_VERSION:
            raise RuntimeError(
                f"{jsonl_path}: readout_prompt_version "
                f"{stored_row.get('readout_prompt_version')!r} != "
                f"{QH_READOUT_PROMPT_VERSION!r}; regenerate this stale artifact"
            )
        if stored_row.get("target_semantic") != expected_semantic:
            raise RuntimeError(
                f"{jsonl_path}: target_semantic {stored_row.get('target_semantic')!r} != "
                f"{expected_semantic!r} — this artifact belongs to a different mode/arm"
            )
    print(f"Q_H drift guard OK: {len(rows)} rebuilt rows match {jsonl_path} exactly.")


def select_qh_siblings(qy_rows, qh_rows):
    """The Q_H rows actually paired with `qy_rows`, in `qy_rows` order.

    This is the single selection used both to build the paired dataset and to describe
    the training set in metadata, so the two can never disagree. It must run AFTER any
    --train-all merge AND any --smoke truncation: those change which rows the Trainer
    consumes, and metadata that ignored the truncation would claim rows that were never
    seen.
    """
    by_qy = {r["qy_row_id"]: r for r in qh_rows}
    if len(by_qy) != len(qh_rows):
        raise RuntimeError("duplicate Q_H sibling row ids")
    try:
        selected = [by_qy[r["row_id"]] for r in qy_rows]
    except KeyError as exc:
        raise RuntimeError(f"no Q_H sibling row for Q_Y row {exc.args[0]!r}") from exc
    pair_qy_qh_rows(qy_rows, selected)
    return selected


def qh_artifact_provenance_metadata(cfg, artifact_train_rows, artifact_eval_rows,
                                    training_rows, trainer_eval_rows, train_all):
    """Return separate artifact-file and consumed-training-set hashes.

    Under ``--train-all``, the consumed training set merges the train and eval artifacts,
    so its digest differs from either file digest. ``training_rows`` and
    ``trainer_eval_rows`` must be the post-merge, post-smoke selection returned by
    ``select_qh_siblings``.
    """
    return {
        "train_artifact": cfg["qh_train_jsonl"],
        "eval_artifact": cfg["qh_eval_jsonl"],
        "train_artifact_rows": len(artifact_train_rows),
        "eval_artifact_rows": len(artifact_eval_rows),
        "train_artifact_sha256": qh_combined_hash(artifact_train_rows),
        "eval_artifact_sha256": qh_combined_hash(artifact_eval_rows),
        "training_set_rows": len(training_rows),
        "training_set_sha256": qh_combined_hash(training_rows),
        "trainer_eval_set_rows": len(trainer_eval_rows),
        "train_all_uses_eval_artifact_as_training_data": bool(train_all),
        "row_provenance_hash_field": QH_ROW_PROVENANCE_HASH_FIELD,
        "combined_hash_policy": (
            "sha256 over the newline-joined per-row "
            f"{QH_ROW_PROVENANCE_HASH_FIELD} chain (whole-row coverage)"
        ),
    }


def print_sample_qh_prompts(rows):
    """One full sample per target letter so the option order is visible to a human."""
    shown = set()
    for row in rows:
        letter = row["target_letter"]
        if letter in shown:
            continue
        shown.add(letter)
        print(f"\n{'=' * 78}\nSAMPLE Q_H ROW {row['row_id']}  (target {letter} = "
              f"{row['target_semantic']}, answer order {row['answer_order']}, "
              f"production order {row['production_answer_order']})\n{'=' * 78}")
        print(row["messages"][0]["content"])
        print(f"--- TARGET LETTER: {letter!r} (scored as {row['target_completion']!r} at the "
              f"final-channel answer slot; NEVER appended to the prompt)")
        if shown == {"A", "B"}:
            break


def build_qh_all(args, write_artifacts):
    """Shared by --check-data/--check-tokenizer/train(): build + check the Q_H rows.

    The builder reloads items, stories, and the split so Q_H artifact construction is
    isolated from the Q_Y-only data path.
    """
    cfg = mode_config(args)
    items = load_items_for_mode(args)
    stories_by_title = load_stories_for_items(items, cfg)
    assert_qh_item_id_positions(items, cfg)
    split = make_split(items)
    prefix = cfg["split_name_prefix"]
    qy_train = build_rows(items, stories_by_title, split["train_indices"], f"{prefix}-train")
    qy_eval = build_rows(items, stories_by_title, split["eval_indices"], f"{prefix}-eval")
    qh_train = build_qh_rows(items, stories_by_title, split["train_indices"],
                             f"{prefix}-train", args)
    qh_eval = build_qh_rows(items, stories_by_title, split["eval_indices"], f"{prefix}-eval", args)

    print(f"Q_H aux ({cfg['dataset']}, mode {cfg['transcript_field']}): target semantic "
          f"{cfg['qh_target_semantic']} (mode lock), schedule {QH_AUX_SCHEDULE_VERSION}, "
          f"readout {QH_READOUT_PROMPT_VERSION}.")
    qh_forbidden_content_checks(qh_train + qh_eval, items, stories_by_title)
    sanity_check_qh_rows(qh_train, qh_eval, qy_train, qy_eval, items, stories_by_title, split)
    audit = qh_alignment_audit(pair_qy_qh_rows(qy_train, qh_train)
                               + pair_qy_qh_rows(qy_eval, qh_eval),
                               dataset=cfg["dataset"])

    if write_artifacts:
        write_jsonl_atomic(cfg["qh_train_jsonl"], qh_train)
        write_jsonl_atomic(cfg["qh_eval_jsonl"], qh_eval)
        print(f"Wrote {cfg['qh_train_jsonl']} ({len(qh_train)} rows), "
              f"{cfg['qh_eval_jsonl']} ({len(qh_eval)} rows).")
    return qh_train, qh_eval, audit


# ---------------------------------------------------------------------------
# Rationale generation + validation (torch-free except the teacher backend)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")  # XML/HTML-like tag; ignores "x < y", "<3"
_HARMONY_RE = re.compile(r"<\|.*?\|>")
_ANSWER_LINE_RE = re.compile(r"(?i)answer:\s*[AB]\b")  # the final-channel verdict form
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_PRIVATE_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)
GOLD_SUBSTR_MIN = 20  # a >=20-char gold-answer window counts as a positive gold reference
RATIONALE_FLUSH_EVERY = 20


def _gold_answer_text(item):
    """The correct Q_Y answer text (Y_true), independent of A/B orientation."""
    return item["Q_Y"]["Y_true"]


def _norm_ws(text):
    return " ".join(text.split())


def _contains_substantial_substring(haystack_low, needle_low, min_len=GOLD_SUBSTR_MIN):
    """True if any contiguous >=min_len-char window of needle (ws-normalized) occurs in
    haystack (ws-normalized). Used as the gold-answer-text grounding fallback."""
    h = _norm_ws(haystack_low)
    n = _norm_ws(needle_low)
    if not n:
        return False
    if len(n) < min_len:
        return len(n) >= H_OVERLAP_MIN_CHARS and n in h
    return any(n[i:i + min_len] in h for i in range(len(n) - min_len + 1))


def _gold_debater_for_row(base_row):
    """Return the original transcript identity defending Y_true, and assert that it
    agrees with the row's current A/B orientation. The identity is visible in the
    verifier prompt; this helper never consults Q_H or either hidden answer."""
    stances = base_row.get("debater_stances")
    if not isinstance(stances, dict):
        raise ValueError(f"{base_row.get('row_id')}: missing debater_stances")
    gold = [name for name, stance in stances.items() if stance == "Y_true"]
    if len(gold) != 1:
        raise ValueError(
            f"{base_row.get('row_id')}: expected exactly one Y_true debater, got {stances!r}"
        )
    gold_debater = gold[0]
    target_letter = base_row.get("target_letter")
    mapping = base_row.get("debater_to_letter")
    if not isinstance(mapping, dict) or mapping.get(gold_debater) != target_letter:
        raise ValueError(
            f"{base_row.get('row_id')}: Y_true debater {gold_debater!r} does not map "
            f"to target letter {target_letter!r}"
        )
    return gold_debater


def _word_trigrams(text):
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return set(zip(words, words[1:], words[2:]))


def _debater_public_text(item, debater):
    """Concatenate one side's public arguments, excluding private <thinking> blocks.
    Private planning is not an eligible condition-fidelity anchor even if malformed
    source transcripts accidentally expose it to the verifier."""
    rounds = item["Q_Y"]["transcript"]["rounds"]
    texts = []
    for round_idx, rnd in enumerate(rounds):
        value = rnd.get(debater)
        if not isinstance(value, str):
            raise ValueError(
                f"dataset item {item.get('dataset_index')} round {round_idx}: "
                f"missing/non-string {debater} argument"
            )
        texts.append(_PRIVATE_THINKING_RE.sub(" ", value))
    return "\n".join(texts)


def _gold_unique_trigram_hits(rationale, item, gold_debater):
    """Exact, conservative proxy for transcript-condition fidelity.

    Eligible anchors must occur in the gold-side public argument but not in the
    opposing argument or the visible question/answer texts. This prevents a generic
    answer summary copied from the options from satisfying the guard. It does not
    claim to measure hidden steering; it only proves that the supervised rationale
    retained some content distinctive to the exact correct-side transcript.
    """
    wrong_debater = "Debater B" if gold_debater == "Debater A" else "Debater A"
    gold_ngrams = _word_trigrams(_debater_public_text(item, gold_debater))
    wrong_ngrams = _word_trigrams(_debater_public_text(item, wrong_debater))
    q_y = item["Q_Y"]
    surface_ngrams = _word_trigrams(
        "\n".join((q_y["question"], q_y["Y_true"], q_y["Y_false"]))
    )
    eligible = gold_ngrams - wrong_ngrams - surface_ngrams
    hits = sorted(_word_trigrams(rationale) & eligible)
    return [" ".join(trigram) for trigram in hits]


def build_teacher_messages(base_row, gold_answer_text):
    """Teacher prompt: an authoritative system message (format) + the student's verifier
    user content verbatim + a grading note (gold answer) + an explicit override of the
    passage's answer-format instructions. Only the student's user content is used; the
    base row's assistant target is never fed. The gold-side debater identity
    is re-derived from the visible row mapping and is never sourced from Q_H."""
    prompt_text = base_row["messages"][0]["content"]
    gold_letter = base_row["target_letter"]
    gold_debater = _gold_debater_for_row(base_row)
    user = prompt_text + TEACHER_INSTRUCTION_TEMPLATE.format(
        letter=gold_letter, answer_text=gold_answer_text, gold_debater=gold_debater)
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]


def validate_rationale(rationale, item, base_row):
    """Hard-fail (ValueError) on an unusable/unsafe rationale; return an audit dict of
    NON-fatal findings (organic H-overlap allowed, short-H not substring-checkable).
    Torch-free; run at generation time AND re-run at load/verify time."""
    prompt_text = base_row["messages"][0]["content"]
    target_letter = base_row["target_letter"]
    gold_debater = _gold_debater_for_row(base_row)
    text = rationale.strip()
    if not text:
        raise ValueError("empty rationale")
    if len(text) < RATIONALE_MIN_CHARS:
        raise ValueError(f"rationale too short ({len(text)} < {RATIONALE_MIN_CHARS} chars)")
    if len(text) > RATIONALE_MAX_CHARS:
        raise ValueError(f"rationale too long ({len(text)} > {RATIONALE_MAX_CHARS} chars)")
    if _HARMONY_RE.search(text):
        raise ValueError("rationale contains a harmony control token '<|...|>'")
    tag = _TAG_RE.search(text)
    if tag:
        raise ValueError(f"rationale contains an XML/HTML-like tag {tag.group(0)!r}")
    if "```" in text:
        raise ValueError("rationale contains a code fence")
    if _ANSWER_LINE_RE.search(text):
        raise ValueError("rationale contains a literal 'Answer: A/B' verdict line "
                         "(the verdict belongs only in the final channel)")

    low = text.lower()
    prompt_low = prompt_text.lower()
    for term in RATIONALE_FORBIDDEN_TERMS_HARD:
        if term in low:
            raise ValueError(f"rationale contains forbidden experiment/label term {term!r}")
    for term in RATIONALE_FORBIDDEN_TERMS_IF_ABSENT_FROM_PROMPT:
        if term in low and term not in prompt_low:
            raise ValueError(f"rationale introduces jargon term {term!r} not present in the "
                             "visible transcript/prompt (likely a leak, not story vocabulary)")

    qh_q = item["Q_H"]["question"].strip()
    if len(qh_q) >= H_OVERLAP_MIN_CHARS and qh_q.lower() in low:
        raise ValueError("rationale contains the hidden Q_H question text")

    # Gold-side grounding. The teacher prompt is forced toward the gold answer, so a missing
    # literal letter is usually a paraphrase ("the first reading is better supported"), NOT a
    # wrong-side argument. Hard-fail ONLY when the rationale affirms the WRONG option's letter
    # without any gold reference (a clear wrong-side rationale); otherwise allow and record a
    # weak-grounding audit note. Gold is orientation-safe (target_letter / Y_true).
    def _letter_ref(letter):
        ltr = re.escape(letter.lower())
        return bool(
            re.search(rf"\b(?:answer|option|choice)\s+\(?{ltr}\)?\b", low)
            or re.search(rf"\banswer\s+is\s+\(?{ltr}\)?\b", low)
            or re.search(rf"\(\s*{ltr}\s*\)", low)
        )
    gl = target_letter.lower()
    wl = "b" if gl == "a" else "a"
    refs_gold = _letter_ref(gl) or _contains_substantial_substring(low, _gold_answer_text(item).lower())
    if _letter_ref(wl) and not refs_gold:
        raise ValueError(f"rationale affirms the wrong option (answer {wl.upper()}) without "
                         f"referencing the gold side (answer {target_letter})")

    # Condition-fidelity guard. This is intentionally about retaining information from
    # the exact visible transcript, not about claiming or testing a Q_H steering effect.
    gold_unique_hits = _gold_unique_trigram_hits(text, item, gold_debater)
    if len(gold_unique_hits) < RATIONALE_MIN_GOLD_UNIQUE_TRIGRAM_HITS:
        raise ValueError(
            "rationale is not condition-faithful enough: retained "
            f"{len(gold_unique_hits)} < {RATIONALE_MIN_GOLD_UNIQUE_TRIGRAM_HITS} "
            "eligible trigrams unique to the gold-side public transcript"
        )

    audit = {
        "organic_h_overlap": [],
        "h_not_substring_checkable": [],
        "weak_gold_grounding": not refs_gold,
        "gold_debater": gold_debater,
        "gold_unique_trigram_hits": gold_unique_hits,
    }
    for field in ("H_true", "H_false"):
        h_text = item["Q_H"][field].strip()
        if len(h_text) < H_OVERLAP_MIN_CHARS:
            audit["h_not_substring_checkable"].append(field)
            continue
        h_low = h_text.lower()
        if h_low in low:
            if h_low in prompt_low:
                audit["organic_h_overlap"].append((field, h_text))  # visible: allow + audit
            else:
                raise ValueError(
                    f"rationale introduces hidden {field} answer text {h_text!r} that is NOT "
                    "present in the visible transcript/prompt (teacher-introduced leak)")
    return audit


def build_rationale_row(base_row, rationale, teacher_messages, model, backend, attempt,
                        temperature, audit):
    """A rationale row = the verbatim base row + the rationale text + provenance/hashes."""
    messages = base_row["messages"]
    target = base_row["target"]
    row = dict(base_row)
    row.update({
        "rationale": rationale,
        "rationale_prompt_version": RATIONALE_PROMPT_VERSION,
        "rationale_model": model,
        "rationale_backend": backend,
        "rationale_created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rationale_attempt": attempt,
        "rationale_temperature": temperature,
        "rationale_prompt_sha256": teacher_prompt_hash(teacher_messages),
        "rationale_condition_fidelity": {
            "gold_debater": audit["gold_debater"],
            "gold_unique_trigram_hit_count": len(audit["gold_unique_trigram_hits"]),
            "gold_unique_trigram_hits": audit["gold_unique_trigram_hits"],
        },
        "rationale_sha256": rationale_text_hash(rationale),
        "supervised_continuation": build_supervised_continuation(rationale, target),
        "rationale_row_sha256": rationale_row_hash(messages, target, rationale),
    })
    return row

_RATIONALE_QUOTE_TAG_RE = re.compile(
    r"</?(?:v_quote|u_quote|quote)>",
    re.IGNORECASE,
)

def _clean_rationale(text):
    text = text.strip()
    text = _RATIONALE_QUOTE_TAG_RE.sub("", text)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _attempt_seed(row_id, attempt):
    digest = hashlib.sha256(f"{row_id}|{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _teacher_api_key(args):
    return os.environ.get(args.teacher_api_key_env) or "EMPTY"


def _load_teacher(args, extract=None, device_index=None, stats=None):
    """Build ``generate(messages, temperature, seed)`` with lazy model imports.

    The API backend uses a dedicated ``debate.ApiModelClient`` generation configuration;
    that client does not forward the supplied sampling seed. The Transformers backend
    loads one gpt-oss replica on ``device_index`` (logical GPU 0 by default), seeds its
    sampling generator, decodes special tokens, and applies ``extract`` when supplied or
    the strict Harmony final-channel extractor otherwise. ``stats`` collects call,
    generated-token, and elapsed-time telemetry when requested.
    """
    backend = args.teacher_backend
    if backend == "api":
        import debate  # debate.py module top is import-safe (yaml config only; no torch)
        lm = {"temperature": TEACHER_GREEDY_TEMPERATURE, "top_p": TEACHER_TOP_P,
              "max_tokens": args.teacher_max_new_tokens, "timeout": TEACHER_TIMEOUT}
        client = debate.ApiModelClient(args.teacher_base_url, _teacher_api_key(args),
                                       args.teacher_model, lm)

        def generate(messages, temperature, seed):
            # ApiModelClient does not forward this seed; deterministic API operation uses
            # the temperature-zero attempt. The Transformers backend honors sampled seeds.
            lm["temperature"] = temperature  # ApiModelClient.generate re-reads self.lm
            started = time.perf_counter()
            raw = client.generate(messages)
            if stats is not None:
                # new_tokens is unavailable over the API; calls/seconds still are.
                stats["calls"] = stats.get("calls", 0) + 1
                stats["seconds"] = stats.get("seconds", 0.0) + (time.perf_counter() - started)
            return extract(raw).strip() if extract is not None else raw.strip()
        return generate

    if backend == "transformers":
        import contextlib
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                                  Mxfp4Config)
        from judge_common import extract_final_channel

        cuda_index = 0 if device_index is None else int(device_index)
        # AutoConfig and AutoTokenizer are CPU-only metadata/vocab loads, so they stay
        # OUTSIDE the device context; only the model materialization needs it.
        cfg = AutoConfig.from_pretrained(args.teacher_model)
        quantized = getattr(cfg, "quantization_config", None) is not None
        load_kwargs = dict(attn_implementation="eager", dtype=torch.bfloat16,
                           device_map={"": cuda_index}, low_cpu_mem_usage=True)
        if quantized:
            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print(f"Loading teacher {args.teacher_model} "
              f"({'MXFP4->bf16 dequantize' if quantized else 'bf16'}) on "
              f"cuda:{cuda_index} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # ``device_map`` places weights, while MXFP4 materialization kernels use the
        # loading thread's active CUDA device. Bind that device for an explicitly pinned
        # in-process replica; isolated worker processes use their sole visible device.
        load_context = (contextlib.nullcontext() if device_index is None
                        else torch.cuda.device(cuda_index))
        with load_context:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs)
            model.eval()
            model.config.use_cache = True

        def generate(messages, temperature, seed):
            if device_index is not None:
                # Only reachable on the in-process pinned path (tests): production
                # children are already narrowed to one GPU by CUDA_VISIBLE_DEVICES.
                torch.cuda.set_device(cuda_index)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            gen_kwargs = {"max_new_tokens": args.teacher_max_new_tokens,
                          "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
            if temperature and temperature > 0:
                if seed is not None:
                    if device_index is None:
                        torch.manual_seed(seed)          # process-local teacher RNG
                    else:
                        # Use the selected device's CUDA RNG for an in-process pinned
                        # replica. Isolated workers seed their process-local RNG above.
                        torch.cuda.manual_seed(seed)
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=TEACHER_TOP_P)
            else:
                gen_kwargs["do_sample"] = False
            started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            new = out[0][inputs["input_ids"].shape[-1]:]
            if stats is not None:
                stats["calls"] = stats.get("calls", 0) + 1
                stats["new_tokens"] = stats.get("new_tokens", 0) + int(new.shape[-1])
                stats["seconds"] = stats.get("seconds", 0.0) + (time.perf_counter() - started)
            decoded = tokenizer.decode(new, skip_special_tokens=False)
            if extract is not None:
                return extract(decoded).strip()
            return extract_final_channel(decoded).strip()
        return generate

    raise RuntimeError(f"unknown --teacher-backend {backend!r}")


def generate_one_rationale(generate_fn, base_row, item, args):
    """Attempt 0 greedy; up to TEACHER_MAX_ATTEMPTS-1 seeded sampled retries on a
    validation/extraction failure. Returns (rationale, teacher_messages, attempt,
    temperature, audit); raises if no attempt yields a valid rationale."""
    teacher_messages = build_teacher_messages(base_row, _gold_answer_text(item))
    last_err = None
    for attempt in range(TEACHER_MAX_ATTEMPTS):
        temperature = TEACHER_GREEDY_TEMPERATURE if attempt == 0 else TEACHER_RETRY_TEMPERATURE
        seed = None if attempt == 0 else _attempt_seed(base_row["row_id"], attempt)
        try:
            rationale = _clean_rationale(generate_fn(teacher_messages, temperature, seed))
        except Exception as exc:  # noqa: BLE001 (record + retry)
            last_err = f"generation error: {exc}"
            continue
        try:
            audit = validate_rationale(rationale, item, base_row)
        except ValueError as exc:
            last_err = str(exc)
            continue
        return rationale, teacher_messages, attempt, temperature, audit
    raise RuntimeError(f"row {base_row['row_id']}: no valid rationale after "
                       f"{TEACHER_MAX_ATTEMPTS} attempts; last failure: {last_err}")


def _rationale_paths(split, args):
    cfg = mode_config(args)
    if split == "train":
        return cfg["train_jsonl"], cfg["rationale_train_jsonl"], f"{cfg['split_name_prefix']}-train"
    if split == "eval":
        return cfg["eval_jsonl"], cfg["rationale_eval_jsonl"], f"{cfg['split_name_prefix']}-eval"
    raise ValueError(f"unknown split {split!r}")


def _load_split_base_rows(split, args):
    """Rebuild ONE split's base rows from source, re-run leak checks, and drift-guard
    against the audited base jsonl. Returns (items, base_rows, rationale_path,
    manifest_path, split_name)."""
    cfg = mode_config(args)
    base_jsonl, rationale_path, split_name = _rationale_paths(split, args)
    manifest_path = cfg["rationale_manifest_template"].format(split=split)
    items = load_items_for_mode(args)
    stories_by_title = load_stories_for_items(items, cfg)
    indices_obj = make_split(items)
    indices = indices_obj["train_indices"] if split == "train" else indices_obj["eval_indices"]
    base_rows = build_rows(items, stories_by_title, indices, split_name)
    forbidden_content_checks(base_rows, items, stories_by_title)
    verify_artifacts_match(base_rows, base_jsonl)
    return items, base_rows, rationale_path, manifest_path, split_name


def _verify_rationale_row(base, rat, item):
    """Raise ValueError unless `rat` (a loaded rationale row) is a valid, bound,
    guardrail-passing rationale for `base` (a freshly rebuilt base row). The training
    surface (messages/target) comes from ``base``; rationale text is accepted from the
    file only after these checks. Returns the audit."""
    if rat.get("row_id") != base["row_id"]:
        raise ValueError(f"row_id mismatch {rat.get('row_id')!r} != {base['row_id']!r}")
    if rat.get("rationale_prompt_version") != RATIONALE_PROMPT_VERSION:
        raise ValueError(
            f"{base['row_id']}: rationale_prompt_version "
            f"{rat.get('rationale_prompt_version')!r} != {RATIONALE_PROMPT_VERSION!r}; "
            "regenerate this stale rationale"
        )
    if rat.get("row_sha256") != base["row_sha256"]:
        raise ValueError(f"{base['row_id']}: base row_sha256 mismatch (prompt/target drift)")
    if rat.get("messages") != base["messages"] or rat.get("target") != base["target"]:
        raise ValueError(f"{base['row_id']}: messages/target differ from the source rebuild")
    rationale = rat.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError(f"{base['row_id']}: missing/non-string rationale")
    if rationale_text_hash(rationale) != rat.get("rationale_sha256"):
        raise ValueError(f"{base['row_id']}: rationale_sha256 mismatch")
    if rationale_row_hash(base["messages"], base["target"], rationale) != rat.get("rationale_row_sha256"):
        raise ValueError(f"{base['row_id']}: rationale_row_sha256 mismatch")
    expected_cont = build_supervised_continuation(rationale, base["target"])
    if rat.get("supervised_continuation") != expected_cont:
        raise ValueError(f"{base['row_id']}: supervised_continuation does not reconstruct")
    # Re-derive the teacher prompt from the rebuilt row and gold answer. Its digest proves
    # the prompt used only Q_Y content, the gold letter, and Y_true—not Q_H.
    teacher_messages = build_teacher_messages(base, _gold_answer_text(item))
    if teacher_prompt_hash(teacher_messages) != rat.get("rationale_prompt_sha256"):
        raise ValueError(f"{base['row_id']}: rationale_prompt_sha256 mismatch "
                         "(teacher prompt drift/tamper, or built from wrong inputs)")
    audit = validate_rationale(rationale, item, base)
    expected_fidelity = {
        "gold_debater": audit["gold_debater"],
        "gold_unique_trigram_hit_count": len(audit["gold_unique_trigram_hits"]),
        "gold_unique_trigram_hits": audit["gold_unique_trigram_hits"],
    }
    if rat.get("rationale_condition_fidelity") != expected_fidelity:
        raise ValueError(
            f"{base['row_id']}: rationale_condition_fidelity does not match "
            "the recomputed transcript-specific audit"
        )
    return audit


def _verify_manifest(manifest_path, rat_rows, split):
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"{manifest_path} not found — (re)generate with "
                           f"--generate-rationales --split {split}.")
    with open(manifest_path) as f:
        man = json.load(f)
    if man.get("rationale_prompt_version") != RATIONALE_PROMPT_VERSION:
        raise RuntimeError(
            f"{manifest_path}: rationale_prompt_version "
            f"{man.get('rationale_prompt_version')!r} != {RATIONALE_PROMPT_VERSION!r}; "
            "regenerate this stale artifact"
        )
    if (man.get("rationale_min_gold_unique_trigram_hits")
            != RATIONALE_MIN_GOLD_UNIQUE_TRIGRAM_HITS):
        raise RuntimeError(
            f"{manifest_path}: condition-fidelity threshold does not match the current code"
        )
    if man.get("split") != split:
        raise RuntimeError(f"manifest split {man.get('split')!r} != {split!r}")
    if man.get("row_count") != len(rat_rows):
        raise RuntimeError(f"manifest row_count {man.get('row_count')} != {len(rat_rows)}")
    expected = combined_rationale_hash(rat_rows)
    if man.get("combined_sha256") != expected:
        raise RuntimeError(f"{manifest_path}: combined_sha256 mismatch — partial / reordered "
                           "/ swapped rationale file vs manifest.")


def load_and_verify_rationale_rows(base_rows, rationale_path, manifest_path, items, split):
    """Load the rationale artifact, verify 1:1 alignment + all hashes + manifest +
    guardrails against the rebuilt base rows, and return {row_id: rationale_text}.
    Raises (no silent regeneration) if the artifact is missing or fails any check."""
    if not os.path.exists(rationale_path):
        raise RuntimeError(
            f"{rationale_path} not found. Rationale targets are generated offline — run "
            f"`--generate-rationales --split {split}` (cluster/GPU) before training. "
            "Training never regenerates rationales online.")
    with open(rationale_path) as f:
        rat_rows = [json.loads(line) for line in f if line.strip()]
    if len(rat_rows) != len(base_rows):
        raise RuntimeError(f"{rationale_path}: {len(rat_rows)} rows != {len(base_rows)} base rows")
    _verify_manifest(manifest_path, rat_rows, split)

    rationales = {}
    organic, not_checkable, weak_gold, fidelity_counts = [], set(), 0, []
    for base, rat in zip(base_rows, rat_rows):
        item = items[base["dataset_index"]]
        audit = _verify_rationale_row(base, rat, item)
        rationales[base["row_id"]] = rat["rationale"]
        for field, txt in audit["organic_h_overlap"]:
            organic.append((base["row_id"], field, txt))
        for field in audit["h_not_substring_checkable"]:
            not_checkable.add((base["dataset_index"], field))
        if audit.get("weak_gold_grounding"):
            weak_gold += 1
        fidelity_counts.append(len(audit["gold_unique_trigram_hits"]))
    print(f"Rationale verify [{split}]: {len(rationales)} rows bound + hashed + guardrail-passed "
          f"(manifest combined hash OK).")
    if weak_gold:
        print(f"  weak gold-grounding (no explicit letter/text ref — paraphrase, allowed): "
              f"{weak_gold}/{len(rationales)}")
    if organic:
        print(f"  organic H-text overlap in rationale (audited, allowed — also in transcript): "
              f"{len(organic)}")
        for rid, field, txt in organic[:10]:
            print(f"    {rid}  {field}={txt!r}")
    if not_checkable:
        print(f"  H-answers not substring-checkable (len < {H_OVERLAP_MIN_CHARS}): "
              f"{len(not_checkable)} (item,field) pairs — leak rule cannot cover these.")
    fidelity_counts.sort()
    print("  condition fidelity (eligible gold-side unique-trigram hits) "
          f"min/median/max={fidelity_counts[0]}/"
          f"{fidelity_counts[len(fidelity_counts) // 2]}/{fidelity_counts[-1]}")
    return rationales


def generate_rationales(args):
    """--generate-rationales --split {train,eval}: build the teacher prompts, generate +
    validate the per-row rationales, write the rationale jsonl + manifest, then re-verify
    the written artifact with the exact training-time loader."""
    split = args.split
    items, base_rows, rationale_path, manifest_path, _ = _load_split_base_rows(split, args)
    base_jsonl, _, _ = _rationale_paths(split, args)
    items_by_index = {r["dataset_index"]: items[r["dataset_index"]] for r in base_rows}

    existing = {}
    if os.path.exists(rationale_path):
        try:
            with open(rationale_path) as f:
                prior = {r["row_id"]: r for r in (json.loads(l) for l in f if l.strip())}
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read existing {rationale_path} ({exc}); regenerating all.")
            prior = {}
        for base in base_rows:
            rat = prior.get(base["row_id"])
            if rat is None:
                continue
            # Reuse only rows produced by the requested teacher and backend; the manifest
            # must not relabel a row from another generation configuration.
            if (rat.get("rationale_model") != args.teacher_model
                    or rat.get("rationale_backend") != args.teacher_backend):
                continue  # different teacher config: regenerate
            try:
                _verify_rationale_row(base, rat, items_by_index[base["dataset_index"]])
            except ValueError:
                continue  # stale/invalid: regenerate
            existing[base["row_id"]] = rat
        print(f"Resume: {len(existing)}/{len(base_rows)} existing rationale rows still valid "
              f"(teacher {args.teacher_model}/{args.teacher_backend}).")

    from tqdm import tqdm  # lazy: only the GPU generate path needs the progress bar

    generate_fn = None  # load the (heavy) teacher only if something needs generating
    out_rows = []
    n_generated = 0
    pbar = tqdm(base_rows, desc=f"generate-rationales[{split}]", unit="row")
    for base in pbar:
        rid = base["row_id"]
        if rid in existing:
            out_rows.append(existing[rid])
            pbar.set_postfix(gen=n_generated, reused=len(existing))
            continue
        if generate_fn is None:
            generate_fn = _load_teacher(args)
        item = items_by_index[base["dataset_index"]]
        rationale, tmsgs, attempt, temperature, audit = generate_one_rationale(
            generate_fn, base, item, args)
        row = build_rationale_row(base, rationale, tmsgs, args.teacher_model,
                                  args.teacher_backend, attempt, temperature, audit)
        out_rows.append(row)
        n_generated += 1
        if attempt:  # surface retries (anomalies) without clobbering the bar
            pbar.write(f"  {rid}: needed retry#{attempt}@T{temperature} -> {len(rationale)} chars")
        pbar.set_postfix(gen=n_generated, reused=len(existing), last_chars=len(rationale))
        if n_generated % RATIONALE_FLUSH_EVERY == 0:
            write_jsonl_atomic(rationale_path, out_rows)  # in base order; safe to resume from

    manifest = {
        "dataset": args.dataset,
        "mode": args.mode,
        "split": split,
        "rationale_prompt_version": RATIONALE_PROMPT_VERSION,
        "row_count": len(out_rows),
        "combined_sha256": combined_rationale_hash(out_rows),
        "base_jsonl": base_jsonl,
        "teacher_model": args.teacher_model,
        "teacher_backend": args.teacher_backend,
        "teacher_max_new_tokens": args.teacher_max_new_tokens,
        "greedy_temperature": TEACHER_GREEDY_TEMPERATURE,
        "retry_temperature": TEACHER_RETRY_TEMPERATURE,
        "max_attempts": TEACHER_MAX_ATTEMPTS,
        "top_p": TEACHER_TOP_P,
        "rationale_min_chars": RATIONALE_MIN_CHARS,
        "rationale_max_chars": RATIONALE_MAX_CHARS,
        "rationale_min_gold_unique_trigram_hits":
            RATIONALE_MIN_GOLD_UNIQUE_TRIGRAM_HITS,
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_jsonl_atomic(rationale_path, out_rows)
    write_json_atomic(manifest_path, manifest)
    print(f"Wrote {rationale_path} ({len(out_rows)} rows; {n_generated} newly generated, "
          f"{len(existing)} reused) + {manifest_path}.")

    # Re-verify the written artifact with the EXACT training-time loader.
    load_and_verify_rationale_rows(base_rows, rationale_path, manifest_path, items, split)
    print(f"--generate-rationales [{split}] complete and re-verified.")


def check_rationales(args):
    """--check-rationales (torch-free): validate BOTH split artifacts with the
    training-time loader (alignment + hashes + manifest + guardrails + balance)."""
    for split in ("train", "eval"):
        items, base_rows, rationale_path, manifest_path, _ = _load_split_base_rows(split, args)
        rationales = load_and_verify_rationale_rows(
            base_rows, rationale_path, manifest_path, items, split)
        letters = Counter(r["target_letter"] for r in base_rows)
        assert len(rationales) == len(base_rows), (len(rationales), len(base_rows))
        assert letters["A"] == letters["B"], f"{split} orientation imbalance: {dict(letters)}"
        print(f"--check-rationales [{split}]: {len(rationales)} rationales, paired-orientation "
              f"balance A={letters['A']} B={letters['B']}.")
    print("--check-rationales complete: both splits verified.")


def resolve_rationales(base_rows, split, items, args):
    """Attach row['rationale'] to each base row: the constant ANALYSIS_TARGET under
    --debug-constant-rationale (smoke / --check-tokenizer plumbing only), else the
    verified offline artifact (load_and_verify_rationale_rows raises if it is missing).
    The answer-only arm returns before resolving any rationale path or manifest."""
    supervision = supervision_of(args)
    if supervision != SUPERVISION_RATIONALE:
        stale = [r["row_id"] for r in base_rows if "rationale" in r]
        if stale:
            raise RuntimeError(
                f"{supervision} rows unexpectedly carry rationale content "
                f"(e.g. {stale[:3]}); refusing to risk teacher-forcing hidden CoT."
            )
        if supervision == SUPERVISION_ANSWER_ONLY:
            print(
                f"[answer-only] {split}: no rationale supervision or artifacts; "
                "supervised span = final-channel scaffold + 'Answer: X' + return token "
                f"({len(base_rows)} rows)."
            )
        else:
            print(f"[{supervision}] {split}: no rationale supervision or artifacts "
                  f"({len(base_rows)} rows); see resolve_grounded().")
        return
    if getattr(args, "debug_constant_rationale", False):
        for r in base_rows:
            r["rationale"] = ANALYSIS_TARGET
        print(f"[debug-constant-rationale] {split}: constant ANALYSIS_TARGET on "
              f"{len(base_rows)} rows (plumbing check only — NOT a real training target).")
        return
    cfg = mode_config(args)
    _, rationale_path, _ = _rationale_paths(split, args)
    manifest_path = cfg["rationale_manifest_template"].format(split=split)
    rationales = load_and_verify_rationale_rows(
        base_rows, rationale_path, manifest_path, items, split)
    for r in base_rows:
        r["rationale"] = rationales[r["row_id"]]


# ---------------------------------------------------------------------------
# Grounded adjudication artifacts (torch-free except the teacher backend)
# ---------------------------------------------------------------------------
# EVERYTHING below runs ONLY under --supervision grounded. The rationale and
# answer-only paths never call it, so their rows/bytes/hashes/artifacts are untouched.

def grounded_teacher_orientation(pair_id):
    """Canonical orientation the TEACHER sees for this item, drawn from a hash.

    The teacher runs ONCE per item (the analysis is orientation-invariant), so some
    orientation must be chosen. Fixing it to A=Y_true would let a model's A-position
    prior inflate blind-tier winner agreement — the "blind" tier would quietly leak the
    gold side. Hashing keeps the choice deterministic and ~50/50 across the pool.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError(f"pair_id must be a non-empty string, got {pair_id!r}")
    key = f"{SPLIT_SEED}|{pair_id}|{GROUNDED_PROMPT_VERSION}"
    return ORIENTATIONS[hashlib.sha256(key.encode("utf-8")).digest()[0] % 2]


def _grounded_round_texts(item, debater):
    """Per-round public argument text of one side, with private <thinking> blocks removed
    AND all quote tags stripped, so a span copied out of the rendered prompt (where quotes
    are re-tagged <v_quote>/<u_quote>) still matches its source."""
    rounds = item["Q_Y"]["transcript"]["rounds"]
    texts = []
    for round_idx, rnd in enumerate(rounds):
        value = rnd.get(debater)
        if not isinstance(value, str):
            raise ValueError(
                f"dataset item {item.get('dataset_index')} round {round_idx}: "
                f"missing/non-string {debater} argument"
            )
        texts.append(_RATIONALE_QUOTE_TAG_RE.sub("", _PRIVATE_THINKING_RE.sub(" ", value)))
    return texts


def _grounded_side_text(item, debater):
    return "\n".join(_grounded_round_texts(item, debater))


def _clean_grounded_output(text):
    """Normalize raw teacher output before the strict parse: strip quote tags, normalize
    newlines, drop blank lines and surrounding whitespace. Nothing else is repaired."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RATIONALE_QUOTE_TAG_RE.sub("", text)
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse_grounded_output(text):
    """Strict RAW-teacher-schema parse (GROUNDED_TEACHER_OUTPUT_SCHEMA) -> dict.

    Deliberately REJECTS the rendered schema (which writes "Debater A's key evidence:"
    and a "(verified)" suffix), so a rendered analysis can never be mistaken for teacher
    output. assert_rendered_analysis_shape() is the mirror check in the other direction.
    """
    if not isinstance(text, str):
        raise ValueError(f"teacher output must be a string, got {type(text).__name__}")
    lines = _clean_grounded_output(text)
    if len(lines) != 4:
        raise ValueError(
            f"expected exactly 4 non-empty lines in the teacher schema, got {len(lines)}"
        )
    evidence = {}
    for idx, letter in enumerate(("A", "B")):
        match = _GROUNDED_EVIDENCE_LINE_RE.match(lines[idx])
        if match is None or match.group(1) != letter:
            raise ValueError(
                f"line {idx + 1} is not a 'Debater {letter} evidence: \"...\"' line: "
                f"{lines[idx][:120]!r}"
            )
        span = match.group(2).strip()
        if not span:
            raise ValueError(f"Debater {letter} evidence span is empty")
        evidence[f"Debater {letter}"] = span
    check_match = _GROUNDED_CHECK_LINE_RE.match(lines[2])
    if check_match is None:
        raise ValueError(f"line 3 is not a 'Check: ...' line: {lines[2][:120]!r}")
    conclusion_match = _GROUNDED_CONCLUSION_LINE_RE.match(lines[3])
    if conclusion_match is None:
        raise ValueError(f"line 4 is not the exact conclusion line: {lines[3][:120]!r}")
    return {
        "evidence": evidence,
        "check_text": check_match.group(1).strip(),
        "winner_debater": conclusion_match.group(1),
    }


def _last_grounded_schema_block(decoded):
    """The LAST four consecutive non-empty lines of `decoded` that parse as the raw
    teacher schema, or "".

    A window rather than a maximal run: reasoning models routinely wrap the block in prose
    ("Let me check..." above, "Hope that helps" below), and those lines are adjacent to it.
    """
    text = decoded.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for start in range(len(lines) - 4, -1, -1):
        window = "\n".join(lines[start:start + 4])
        try:
            parse_grounded_output(window)
        except ValueError:
            continue
        return window
    return ""


def extract_grounded_teacher_output(decoded):
    """Extract one strictly validated grounded block from a teacher completion.

    Search, in order, the Harmony final channel, the analysis channel, and the last
    schema-shaped four-line window in the raw completion. Return the first candidate that
    passes ``parse_grounded_output``. If none passes, return the first non-empty candidate
    so the audit records an informative failure reason.
    """
    from judge_common import extract_analysis_channel, extract_final_channel

    candidates = [
        (extract_final_channel(decoded) or "").strip(),
        (extract_analysis_channel(decoded) or "").strip(),
        _last_grounded_schema_block(decoded),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parse_grounded_output(candidate)
        except ValueError:
            continue
        return candidate
    return next((c for c in candidates if c), "")


def render_grounded_analysis(evidence, check_text, winner_debater):
    """Render the supervised analysis G. The HARNESS owns these bytes: the
    (verified)/(unverified) annotations come from the recomputed status in `evidence`,
    never from the teacher, and the debater-name space makes G orientation-invariant."""
    if winner_debater not in GROUNDED_DEBATERS:
        raise ValueError(f"unknown winner {winner_debater!r}")
    lines = []
    for debater in GROUNDED_DEBATERS:
        entry = evidence[debater]
        status = "verified" if entry["verified"] else "unverified"
        lines.append(f'{debater}\'s key evidence: "{entry["span"]}" ({status})')
    lines.append(f"Check: {check_text}")
    lines.append(f"Conclusion: {winner_debater}{GROUNDED_CONCLUSION_SUFFIX}")
    return "\n".join(lines)


def assert_rendered_analysis_shape(analysis):
    """Shape check for the RENDERED schema (GROUNDED_ANALYSIS_SCHEMA).

    Rejects the raw teacher schema, so the two formats can never be confused. Returns the
    parsed (evidence-span, status) pairs for cross-checking."""
    if not isinstance(analysis, str):
        raise ValueError(f"analysis must be a string, got {type(analysis).__name__}")
    lines = analysis.split("\n")
    if len(lines) != 4:
        raise ValueError(f"rendered analysis must be exactly 4 lines, got {len(lines)}")
    parsed = []
    for idx, debater in enumerate(GROUNDED_DEBATERS):
        match = _GROUNDED_RENDERED_EVIDENCE_RE.match(lines[idx])
        if match is None or f"Debater {match.group(1)}" != debater:
            raise ValueError(
                f"rendered line {idx + 1} is not a \"{debater}'s key evidence: ... "
                f"(verified|unverified)\" line: {lines[idx][:120]!r}"
            )
        parsed.append((match.group(2), match.group(3)))
    if _GROUNDED_CHECK_LINE_RE.match(lines[2]) is None:
        raise ValueError(f"rendered line 3 is not a 'Check: ...' line: {lines[2][:120]!r}")
    winner = None
    for debater in GROUNDED_DEBATERS:
        if lines[3] == f"Conclusion: {debater}{GROUNDED_CONCLUSION_SUFFIX}":
            winner = debater
    if winner is None:
        raise ValueError(f"rendered line 4 is not the exact conclusion line: {lines[3][:120]!r}")
    return parsed, winner


def _grounded_letter_violation(text):
    """First answer-letter reference in `text`, or None. The letter class is UPPERCASE on
    purpose: 'answer a reader expects' must not trip; 'answer A'/'Option B'/'(A)' must."""
    for pattern in GROUNDED_LETTER_BAN_RES:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def grounded_span_audit(span, item, debater, story_norm):
    """Hard-fail unless `span` is a usable, side-attributable, specific verbatim span of
    `debater`'s public arguments; return the recorded evidence entry.

    `verified` is RECOMPUTED here with the verify_quotes predicate (normalized-substring
    against the hidden story) and `round_idx` is DERIVED (0-based) — neither is ever
    trusted from the teacher. Containment must hold inside ONE round, which also blocks
    cross-round splices that the whitespace-collapsed whole-side text would accept.
    """
    from judge_common import normalize_text

    norm = normalize_text(span)
    if len(norm) < GROUNDED_SPAN_MIN_NORM_CHARS:
        raise ValueError(
            f"{debater} span too short ({len(norm)} < {GROUNDED_SPAN_MIN_NORM_CHARS} "
            "normalized chars)"
        )
    if len(norm) > GROUNDED_SPAN_MAX_NORM_CHARS:
        raise ValueError(
            f"{debater} span too long ({len(norm)} > {GROUNDED_SPAN_MAX_NORM_CHARS} "
            "normalized chars)"
        )
    hits = [
        idx for idx, text in enumerate(_grounded_round_texts(item, debater))
        if norm in normalize_text(text)
    ]
    if not hits:
        raise ValueError(
            f"{debater} span is not a verbatim substring of a single round of that "
            "debater's public arguments"
        )
    other = "Debater B" if debater == "Debater A" else "Debater A"
    if norm in normalize_text(_grounded_side_text(item, other)):
        raise ValueError(
            f"{debater} span also occurs in {other}'s arguments; it does not attribute "
            "evidence to a side"
        )
    q_y = item["Q_Y"]
    for field in ("question", "Y_true", "Y_false"):
        if norm in normalize_text(q_y[field]):
            raise ValueError(
                f"{debater} span is text from the {field}; option/question echo is not "
                "transcript evidence"
            )
    return {
        "span": span,
        "round_idx": hits[0],
        "round_idx_ambiguous": len(hits) > 1,
        "verified": norm in story_norm,
        "span_norm_chars": len(norm),
    }


def validate_grounded(parsed, item, base_row, story):
    """Hard-fail (ValueError) on an unusable grounded candidate; return
    (evidence, check_text, winner_debater, analysis, audit).

    H-HYGIENE (see GROUNDED_H_HYGIENE_NOTE): Q_H/H_true/H_false never reach the teacher
    and never enter a supervised field. The H reads at the end of this function are a
    REJECT-ONLY predicate — they can hard-fail a candidate or record an organic overlap,
    and can never add H-derived content to the training signal.
    """
    from judge_common import normalize_text

    prompt_text = base_row["messages"][0]["content"]
    prompt_low = prompt_text.lower()
    story_norm = normalize_text(story)
    check_text = parsed["check_text"].strip()
    winner = parsed["winner_debater"]

    gold_debater = _gold_debater_for_row(base_row)
    if winner != gold_debater:
        raise ValueError(
            f"conclusion names {winner!r} but the Y_true-side debater is {gold_debater!r}"
        )

    evidence = {}
    for debater in GROUNDED_DEBATERS:
        evidence[debater] = grounded_span_audit(
            parsed["evidence"][debater], item, debater, story_norm
        )
    if (normalize_text(evidence["Debater A"]["span"])
            == normalize_text(evidence["Debater B"]["span"])):
        raise ValueError("both debaters' evidence spans normalize to the same text")

    if '"' in check_text:
        raise ValueError("check text contains a double-quote character")
    if len(check_text) < GROUNDED_CHECK_MIN_CHARS:
        raise ValueError(
            f"check text too short ({len(check_text)} < {GROUNDED_CHECK_MIN_CHARS} chars)"
        )
    if len(check_text) > GROUNDED_CHECK_MAX_CHARS:
        raise ValueError(
            f"check text too long ({len(check_text)} > {GROUNDED_CHECK_MAX_CHARS} chars)"
        )
    sentences = [s for s in _GROUNDED_SENTENCE_SPLIT_RE.split(check_text) if s.strip()]
    if not GROUNDED_CHECK_MIN_SENTENCES <= len(sentences) <= GROUNDED_CHECK_MAX_SENTENCES:
        raise ValueError(
            f"check text has {len(sentences)} sentences; expected "
            f"{GROUNDED_CHECK_MIN_SENTENCES}-{GROUNDED_CHECK_MAX_SENTENCES}"
        )
    for debater in GROUNDED_DEBATERS:
        if debater not in check_text:
            raise ValueError(f"check text does not name {debater}; it is not a comparison")
        if normalize_text(check_text) == normalize_text(evidence[debater]["span"]):
            raise ValueError(f"check text is just a copy of {debater}'s evidence span")

    analysis = render_grounded_analysis(evidence, check_text, winner)
    assert_rendered_analysis_shape(analysis)

    if _HARMONY_RE.search(analysis):
        raise ValueError("grounded analysis contains a harmony control token '<|...|>'")
    tag = _TAG_RE.search(analysis)
    if tag:
        raise ValueError(f"grounded analysis contains an XML/HTML-like tag {tag.group(0)!r}")
    if "```" in analysis:
        raise ValueError("grounded analysis contains a code fence")
    letter = _grounded_letter_violation(analysis)
    if letter is not None:
        raise ValueError(
            f"grounded analysis contains an answer-letter reference {letter!r}; the "
            "analysis must be orientation-invariant debater-name space"
        )
    if len(analysis) < GROUNDED_MIN_CHARS:
        raise ValueError(
            f"grounded analysis too short ({len(analysis)} < {GROUNDED_MIN_CHARS} chars)"
        )
    if len(analysis) > GROUNDED_MAX_CHARS:
        raise ValueError(
            f"grounded analysis too long ({len(analysis)} > {GROUNDED_MAX_CHARS} chars)"
        )

    low = analysis.lower()
    for term in GROUNDED_FORBIDDEN_TERMS_HARD:
        if term in low:
            raise ValueError(f"grounded analysis contains forbidden term {term!r}")
    for term in GROUNDED_FORBIDDEN_TERMS_IF_ABSENT_FROM_PROMPT:
        if term in low and term not in prompt_low:
            raise ValueError(
                f"grounded analysis introduces jargon term {term!r} not present in the "
                "visible transcript/prompt"
            )

    # ---- reject-only H predicate; never a source of supervised content ----
    qh_question = item["Q_H"]["question"].strip()
    if len(qh_question) >= H_OVERLAP_MIN_CHARS and qh_question.lower() in low:
        raise ValueError("grounded analysis contains the hidden Q_H question text")
    organic, not_checkable = [], []
    for field in ("H_true", "H_false"):
        h_text = item["Q_H"][field].strip()
        if len(h_text) < H_OVERLAP_MIN_CHARS:
            not_checkable.append(field)
            continue
        h_low = h_text.lower()
        if h_low in low:
            if h_low in prompt_low:
                organic.append([field, h_text])  # visible in the transcript: allow + record
            else:
                raise ValueError(
                    f"grounded analysis introduces hidden {field} answer text {h_text!r} "
                    "that is NOT present in the visible transcript/prompt"
                )

    audit = {
        "organic_h_overlap": organic,
        "h_not_substring_checkable": not_checkable,
        "check_chars": len(check_text),
        "check_sentences": len(sentences),
        "letters_clean": True,
        "spans_distinct": True,
        "winner_stance": "Y_true",
    }
    return evidence, check_text, winner, analysis, audit


def grounded_text_hash(analysis):
    """sha256 of the rendered analysis (utf-8)."""
    return hashlib.sha256(analysis.encode("utf-8")).hexdigest()


def grounded_row_hash(messages, target, analysis):
    """Binds the analysis to the exact canonical prompt+target it was written against."""
    canonical = json.dumps([messages, target, analysis], sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grounded_content_hash(record):
    """DETERMINISTIC content digest over GROUNDED_CONTENT_FIELDS, in a fixed field order.

    The exact key-set assertion keeps whole-record coverage (a field added later cannot
    dodge the digest — it fails loudly until it is classified as content or volatile),
    while GROUNDED_VOLATILE_FIELDS keeps runtime-only values (created_at) out, so a
    semantically identical regeneration hashes identically.
    """
    missing = [f for f in GROUNDED_RECORD_FIELDS if f not in record]
    unexpected = sorted(set(record) - set(GROUNDED_RECORD_FIELDS))
    if missing or unexpected:
        raise ValueError(
            f"grounded record {record.get('record_id')!r} key set mismatch: "
            f"missing={missing}, unexpected={unexpected}. Every record must carry exactly "
            "GROUNDED_RECORD_FIELDS so the content digest cannot silently lose coverage."
        )
    payload = [[field, record[field]] for field in GROUNDED_CONTENT_FIELDS]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def grounded_combined_hash(records):
    """Ordered chain over the deterministic per-record content digests."""
    joined = "\n".join(r["grounded_content_sha256"] for r in records)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_grounded_teacher_messages(base_row, saw_gold, gold_answer_text=None,
                                    gold_debater=None):
    """Teacher prompt = the student's exact verifier user content + the adjudication
    instruction, plus (gold tier only) a note naming the gold ANSWER TEXT and DEBATER —
    never a letter, since the analysis must stay orientation-invariant. Q_H never
    appears."""
    user = base_row["messages"][0]["content"] + GROUNDED_TEACHER_INSTRUCTION
    if saw_gold:
        if not gold_answer_text or not gold_debater:
            raise RuntimeError("gold-tier teacher prompt requires the answer text + debater")
        user += GROUNDED_GOLD_NOTE_TEMPLATE.format(
            answer_text=gold_answer_text, gold_debater=gold_debater)
    return [
        {"role": "system", "content": GROUNDED_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_grounded_record(item, rows_by_orientation, canonical_row, split_name, cfg, mode,
                          status, teacher_model, teacher_backend, teacher_messages=None,
                          attempt=None, temperature=None, saw_gold=False, evidence=None,
                          check_text=None, winner_debater=None, analysis=None, audit=None,
                          failures=None):
    """One item-level grounded record carrying exactly GROUNDED_RECORD_FIELDS."""
    record = {
        "record_id": item["pair_id"],
        "pair_id": item["pair_id"],
        "dataset_index": item["dataset_index"],
        "source_dataset_index": item.get("source_dataset_index"),
        "story_title": item["story_title"],
        "split": split_name,
        "dataset": cfg["dataset"],
        "mode": mode,
        "grounded_prompt_version": GROUNDED_PROMPT_VERSION,
        "status": status,
        "teacher_row_id": canonical_row["row_id"],
        "teacher_orientation": canonical_row["orientation"],
        "teacher_saw_gold": bool(saw_gold),
        "teacher_tier": ("gold" if saw_gold else "blind") if status == "resolved" else None,
        "teacher_attempt": attempt,
        "teacher_temperature": temperature,
        "teacher_model": teacher_model,
        "teacher_backend": teacher_backend,
        "teacher_prompt_sha256": (teacher_prompt_hash(teacher_messages)
                                  if teacher_messages is not None else None),
        "evidence": evidence,
        "check_text": check_text,
        "winner_debater": winner_debater,
        "analysis": analysis,
        "analysis_chars": len(analysis) if analysis is not None else None,
        "audit": audit,
        "audit_thresholds": grounded_audit_thresholds(),
        "base_row_sha256": {o: rows_by_orientation[o]["row_sha256"] for o in ORIENTATIONS},
        "grounded_sha256": grounded_text_hash(analysis) if analysis is not None else None,
        "grounded_row_sha256": (
            grounded_row_hash(canonical_row["messages"], canonical_row["target"], analysis)
            if analysis is not None else None
        ),
        "failures": list(failures or []),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grounded_content_sha256": None,
    }
    record["grounded_content_sha256"] = grounded_content_hash(record)
    return record


def grounded_attempt_seed(row_id, attempt, epoch=0):
    """Return the deterministic seed for one sampled grounded attempt.

    Epoch zero uses the row/attempt identity. Retrying an unresolved record increments the
    epoch so sampled attempts receive a new deterministic draw; temperature-zero attempts
    remain unaffected.
    """
    if epoch:
        return _attempt_seed(f"{row_id}|grounded-retry-{epoch}", attempt)
    return _attempt_seed(row_id, attempt)


def grounded_next_epoch(record):
    """Retry epoch for a prior record: one past the highest epoch its failures recorded.

    Fresh items and regeneration of a RESOLVED record use epoch 0; every retry of an
    unresolved record increments, so repeated re-runs keep drawing new samples.
    """
    if not record or record.get("status") == "resolved":
        return 0
    epochs = [int(f.get("epoch", 0)) for f in (record.get("failures") or [])]
    return 1 + max(epochs, default=0)


def generate_one_grounded(generate_fn, base_row, item, story, args, epoch=0):
    """Two BLIND attempts (greedy, then seeded sample) followed by two GOLD-NOTE attempts.

    Blind attempts are accepted only if every audit passes AND the conclusion names the
    Y_true-side debater — a STaR-style correctness filter that removes the label leak at
    source for the accepting majority. `epoch` salts the sampled-attempt seeds so a retry
    of an unresolved item draws differently (see grounded_attempt_seed). Returns
    (result or None, failures); every failure entry records its attempt, tier and epoch."""
    gold_answer_text = _gold_answer_text(item)
    gold_debater = _gold_debater_for_row(base_row)
    failures = []
    for attempt in range(GROUNDED_MAX_ATTEMPTS):
        saw_gold = attempt >= GROUNDED_BLIND_ATTEMPTS
        tier = "gold" if saw_gold else "blind"
        tier_index = attempt - GROUNDED_BLIND_ATTEMPTS if saw_gold else attempt
        temperature = (TEACHER_GREEDY_TEMPERATURE if tier_index == 0
                       else TEACHER_RETRY_TEMPERATURE)
        seed = (None if tier_index == 0
                else grounded_attempt_seed(base_row["row_id"], attempt, epoch))
        messages = build_grounded_teacher_messages(
            base_row, saw_gold, gold_answer_text, gold_debater)
        try:
            raw = generate_fn(messages, temperature, seed)
        except Exception as exc:  # noqa: BLE001 (record + retry)
            failures.append({"attempt": attempt, "tier": tier, "epoch": epoch,
                             "reason": f"generation error: {exc}"})
            continue
        try:
            parsed = parse_grounded_output(raw)
            evidence, check_text, winner, analysis, audit = validate_grounded(
                parsed, item, base_row, story)
        except ValueError as exc:
            failures.append({"attempt": attempt, "tier": tier, "epoch": epoch,
                             "reason": str(exc)})
            continue
        return {
            "teacher_messages": messages,
            "attempt": attempt,
            "temperature": temperature,
            "saw_gold": saw_gold,
            "evidence": evidence,
            "check_text": check_text,
            "winner_debater": winner,
            "analysis": analysis,
            "audit": audit,
        }, failures
    return None, failures


def _grounded_openai_client(base_url, api_key):
    """The OpenAI client for the grounded API teacher. Indirected through a function so
    tests can inject a fake without importing `openai`.

    max_retries=0 is DELIBERATE: this arm owns its retry policy (see _GroundedApiClient), so
    no transport retry can hide from the artifact's attempt accounting.
    """
    from openai import OpenAI  # lazy: torch-free entry points never import the openai stack

    return OpenAI(base_url=base_url, api_key=api_key, max_retries=0)


def _grounded_is_retryable(exc):
    """Transient transport/server failures only; any other 4xx is fatal on the first
    response, so e.g. a max_model_len 400 lands in `failures` with the server's own message
    instead of being retried four times."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in GROUNDED_RETRYABLE_EXC_NAMES


def _grounded_message_field(message, field):
    """Read `field` off a chat message, tolerating SDK model / model_extra / plain dict.

    `reasoning_content` is a vLLM extension, so depending on the SDK version it can arrive
    as an attribute, inside model_extra, or (under a fake/mapping) as a dict key.
    """
    value = getattr(message, field, None)
    if value:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get(field):
        return extra[field]
    if isinstance(message, dict) and message.get(field):
        return message[field]
    return None


def _grounded_raw_completion(message):
    """Rebuild Harmony channels from an OpenAI-shaped chat message.

    API analysis text may arrive in ``reasoning_content`` rather than ``content``. Mark
    both fields with their channel boundaries so grounded extraction applies the same
    final-channel, analysis-channel, and raw-window priority as the Transformers path.
    """
    content = _grounded_message_field(message, "content") or ""
    reasoning = _grounded_message_field(message, "reasoning_content") or ""
    if not reasoning:
        return content
    raw = f"{ANALYSIS_MARKER}{reasoning}<|end|>"
    if content:
        raw += f"{FINAL_MARKER}{content}<|return|>"
    return raw


class _GroundedApiClient:
    """Seeded, bounded-retry OpenAI-compatible client for the GROUNDED arm only.

    Deliberately NOT debate.ApiModelClient (which the rationale arm and production
    adversarial generation both use, and which is left untouched):
      * it drops the seed, so grounded sampled retries would ignore the retry-epoch seed
        progression that makes "unresolved items are always retried" a real guarantee;
      * it has no retry policy for a multi-hour run;
      * its `generate` re-reads a MUTABLE lm dict, which is a race the moment two threads
        share one client. Here temperature and seed are per-CALL arguments, so there is no
        shared mutable sampling state at all.

    Retries here are TRANSPORT retries inside one ladder attempt: they never append to
    `failures`, never advance attempt/tier/temperature/seed, and never add a telemetry call.
    A seeded request replays the same draw, so a retry is not a second sample.
    """

    def __init__(self, base_url, api_key, model, max_tokens,
                 timeout=GROUNDED_API_TIMEOUT,
                 transport_retries=GROUNDED_API_TRANSPORT_RETRIES,
                 backoff=GROUNDED_API_RETRY_BACKOFF, sleep=None, rng=None):
        self._client = _grounded_openai_client(base_url, api_key)
        self.base_url = base_url
        self.model = model
        self.max_tokens = int(max_tokens)
        self.timeout = timeout
        self.transport_retries = int(transport_retries)
        self.backoff = float(backoff)
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random(0)

    def complete(self, messages, temperature, seed):
        """One completion. Returns (raw harmony-ish text, completion_tokens or None)."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": TEACHER_TOP_P,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if seed is not None:
            # Greedy attempt 0 passes seed=None and sends NO seed key; every sampled attempt
            # sends grounded_attempt_seed(row_id, attempt, epoch) so vLLM seeds that
            # request's sampler and the retry-epoch progression means the same thing on
            # both backends.
            kwargs["seed"] = seed
        for attempt in range(self.transport_retries + 1):
            try:
                completion = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - re-raised below when fatal
                if attempt >= self.transport_retries or not _grounded_is_retryable(exc):
                    raise
                self._sleep((self.backoff ** attempt) * (1.0 + self._rng.random()))
        message = completion.choices[0].message
        usage = getattr(completion, "usage", None)
        new_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        return _grounded_raw_completion(message), new_tokens


def _load_grounded_api_teacher(args, stats=None):
    """generate(messages, temperature, seed) -> grounded block, over HTTP.

    Uses the same closure signature as the local teachers. ``stats`` records one call per
    ladder attempt rather than per transport retry; elapsed seconds include retry backoff,
    and generated-token counts come from API completion usage when available.
    """
    client = _GroundedApiClient(args.teacher_base_url, _teacher_api_key(args),
                               args.teacher_model, args.teacher_max_new_tokens)

    def generate(messages, temperature, seed):
        started = time.perf_counter()
        raw, new_tokens = client.complete(messages, temperature, seed)
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["seconds"] = stats.get("seconds", 0.0) + (time.perf_counter() - started)
            if new_tokens is not None:
                stats["new_tokens"] = stats.get("new_tokens", 0) + int(new_tokens)
        return extract_grounded_teacher_output(raw).strip()

    return generate


def _load_grounded_teacher(args, device_index=None, stats=None):
    """Teacher for the grounded arm: tolerant extraction, optionally pinned to one GPU.

    Single monkeypatch point for tests, and the only place the grounded arm loads a model.
    The api backend goes to the grounded-only seeded client above; the transformers backend
    keeps the shared _load_teacher path byte-for-byte (device_index only ever applies there,
    and the api backend needs no local device).
    """
    if args.teacher_backend == "api":
        return _load_grounded_api_teacher(args, stats=stats)
    return _load_teacher(args, extract=extract_grounded_teacher_output,
                         device_index=device_index, stats=stats)


def _probe_grounded_endpoint(base_url, model, timeout_s=GROUNDED_ENDPOINT_PROBE_TIMEOUT):
    """GET {base_url}/models and confirm `model` is served, EXACTLY.

    Exact match, not debate-bok's substring match: teacher_model is a provenance field baked
    into every record's content hash, so a near-miss must fail loudly rather than be
    recorded. Module level so tests can monkeypatch it (same indirection as
    grounded_mp_context). Only called when API generation has pending work.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"grounded teacher endpoint {url} is not reachable "
            f"({type(exc).__name__}: {exc}). Start the server first, e.g.\n"
            f"  vllm serve <abs path to {BF16_DIR}> --served-model-name {model} "
            "--host 127.0.0.1 --port 18888 --tensor-parallel-size 4 "
            "--max-model-len 16384 --enable-chunked-prefill --dtype bfloat16"
        ) from exc
    served = [entry.get("id") for entry in (payload.get("data") or [])]
    if model not in served:
        raise RuntimeError(
            f"grounded teacher endpoint {url} serves {served!r}, which does not include "
            f"--teacher-model {model!r}. teacher_model is provenance recorded in every "
            "record, so a near-miss is refused: pass `--served-model-name "
            f"{model}` to vllm serve, or point --teacher-model at what is actually served."
        )
    return served


def grounded_mp_context():
    """The multiprocessing context for the teacher pool: ALWAYS 'spawn'.

    Never fork: a forked child inherits the parent's address space, and forking a process
    that has initialised CUDA yields an unusable context in the child. Spawn re-execs a
    clean interpreter, which is also what lets the child set CUDA_VISIBLE_DEVICES before
    torch is ever imported. Safe for this module because its top level is stdlib +
    judge_common + adversarial_transcript (no torch/transformers import at module scope)
    and it is guarded by `if __name__ == "__main__"`, so the re-import is cheap and does
    not re-run main(). Indirected through a function so tests can inject a fake context.
    """
    import multiprocessing

    return multiprocessing.get_context("spawn")


_GROUNDED_VISIBLE_UNSET = object()


def grounded_child_visible_device(device_index, parent_visible=_GROUNDED_VISIBLE_UNSET):
    """The CUDA_VISIBLE_DEVICES value for worker `device_index`.

    Maps THROUGH the parent's list when the scheduler set one: with
    CUDA_VISIBLE_DEVICES="2,3" worker 1 must get "3", not "1". Falls back to the raw index
    when the parent exposes every device.

    `parent_visible` uses a SENTINEL, not None, for "not supplied": None is a legitimate
    snapshot value ("the parent had no list"), and conflating the two would make this
    re-read os.environ after a child had already rewritten it in-process.
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES")
           if parent_visible is _GROUNDED_VISIBLE_UNSET else parent_visible)
    if raw is None or not str(raw).strip():
        return str(device_index)
    visible = [part.strip() for part in str(raw).split(",") if part.strip()]
    if device_index >= len(visible):
        raise RuntimeError(
            f"worker {device_index} has no entry in CUDA_VISIBLE_DEVICES={raw!r} "
            f"({len(visible)} device(s) visible)"
        )
    return visible[device_index]


def _grounded_child_main(payload, task_q, result_q):
    """Entry point of ONE spawned teacher worker. Module-level so it pickles by name.

    Protocol on result_q:
        {"kind": "fatal",  "device_index", "traceback"}    -- died before ready
        {"kind": "ready",  "device_index", "load_seconds"}
        {"kind": "record", "device_index", "slot", "record", "resolved", "telemetry"}
        {"kind": "error",  "device_index", "slot", "traceback"}
    A None on task_q is the shutdown sentinel.
    """
    device_index = payload["device_index"]
    try:
        # Restrict visibility before importing torch so each child sees its assigned GPU
        # as logical device 0 and uses the same single-replica loading path.
        os.environ["CUDA_VISIBLE_DEVICES"] = payload["visible_device"]
        args = argparse.Namespace(**payload["args"])
        cfg = payload["cfg"]
        split_name = payload["split_name"]
        stats = {"calls": 0, "new_tokens": 0, "seconds": 0.0}
        started = time.perf_counter()
        generate_fn = _load_grounded_teacher(args, device_index=None, stats=stats)
        load_seconds = time.perf_counter() - started
    except BaseException:  # noqa: BLE001 - a startup failure must never hang the parent
        import traceback
        result_q.put({"kind": "fatal", "device_index": device_index,
                      "traceback": traceback.format_exc()})
        return
    result_q.put({"kind": "ready", "device_index": device_index,
                  "load_seconds": load_seconds})

    while True:
        task = task_q.get()
        if task is None:
            return
        slot = task["slot"]
        before = dict(stats)
        try:
            record, resolved = _build_grounded_record_for_task(
                task, generate_fn, args, cfg, split_name)
        except BaseException:  # noqa: BLE001 - report, never die silently
            import traceback
            result_q.put({"kind": "error", "device_index": device_index, "slot": slot,
                          "traceback": traceback.format_exc()})
            return
        result_q.put({
            "kind": "record", "device_index": device_index, "slot": slot,
            "record": record, "resolved": resolved,
            "telemetry": {key: stats[key] - before.get(key, 0)
                          for key in ("calls", "new_tokens", "seconds")},
        })


def _build_grounded_record_for_task(task, generate_fn, args, cfg, split_name):
    """The FULL per-item tier ladder + record construction. Shared verbatim by the
    single-worker in-process path and every spawned child, so the two cannot drift."""
    result, failures = generate_one_grounded(
        generate_fn, task["canonical_row"], task["item"], task["story"], args,
        epoch=task["epoch"])
    common = dict(split_name=split_name, cfg=cfg, mode=args.mode,
                  teacher_model=args.teacher_model,
                  teacher_backend=args.teacher_backend)
    if result is None:
        return build_grounded_record(
            task["item"], task["rows"], task["canonical_row"], status="unresolved",
            failures=failures, **common), False
    return build_grounded_record(
        task["item"], task["rows"], task["canonical_row"], status="resolved",
        teacher_messages=result["teacher_messages"], attempt=result["attempt"],
        temperature=result["temperature"], saw_gold=result["saw_gold"],
        evidence=result["evidence"], check_text=result["check_text"],
        winner_debater=result["winner_debater"], analysis=result["analysis"],
        audit=result["audit"], failures=failures, **common), True


class _GroundedProgress:
    """Honest generation telemetry.

    The clock starts AFTER every worker is ready and immediately BEFORE the first task is
    dispatched: model-load seconds are excluded (reported separately) but ALL generation
    latency is included, the first item's included. Throughput and ETA are computed over
    PENDING items only — reused items cost no generation and would inflate both.
    """

    def __init__(self, pending_total, reused_total):
        self.pending_total = pending_total
        self.reused_total = reused_total
        self.pending_done = 0
        self.resolved = 0
        self.unresolved = 0
        self.attempts = 0
        self.new_tokens = 0
        self.generate_calls = 0
        self.generate_seconds = 0.0   # summed teacher-generate time across all workers
        self.load_seconds = {}
        self.started = None

    def start(self):
        self.started = time.perf_counter()

    def record_ready(self, device_index, load_seconds):
        self.load_seconds[device_index] = float(load_seconds)

    def record_item(self, record, telemetry=None):
        self.pending_done += 1
        if record.get("status") == "resolved":
            self.resolved += 1
            self.attempts += int(record.get("teacher_attempt") or 0) + 1
        else:
            self.unresolved += 1
            self.attempts += len(record.get("failures") or [])
        if telemetry:
            self.new_tokens += int(telemetry.get("new_tokens") or 0)
            self.generate_calls += int(telemetry.get("calls") or 0)
            self.generate_seconds += float(telemetry.get("seconds") or 0.0)

    @property
    def elapsed(self):
        return 0.0 if self.started is None else time.perf_counter() - self.started

    def items_per_second(self):
        elapsed = self.elapsed
        return (self.pending_done / elapsed) if elapsed > 0 and self.pending_done else 0.0

    def tokens_per_second(self):
        elapsed = self.elapsed
        return (self.new_tokens / elapsed) if elapsed > 0 and self.new_tokens else 0.0

    def busy_fraction(self):
        """Summed teacher-generate seconds / wall seconds.

        With N workers this approaches N when all teachers are busy and approaches one
        when generation is effectively serial.
        """
        elapsed = self.elapsed
        return (self.generate_seconds / elapsed) if elapsed > 0 else 0.0

    def eta_seconds(self):
        rate = self.items_per_second()
        if rate <= 0:
            return None
        return (self.pending_total - self.pending_done) / rate

    def postfix(self):
        eta = self.eta_seconds()
        return {
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "it/s": f"{self.items_per_second():.3f}",
            "tok/s": f"{self.tokens_per_second():.0f}",
            "eta": "?" if eta is None else f"{eta / 60:.1f}m",
        }

    def summary(self):
        loads = ", ".join(f"gpu{idx}={secs:.0f}s"
                          for idx, secs in sorted(self.load_seconds.items()))
        return (
            f"generation telemetry: {self.pending_done}/{self.pending_total} pending items "
            f"in {self.elapsed:.0f}s ({self.items_per_second():.3f} item/s over PENDING "
            f"only; {self.reused_total} reused item(s) excluded); {self.attempts} teacher "
            f"attempt(s), {self.generate_calls} generate call(s) totalling "
            f"{self.generate_seconds:.0f}s of teacher time "
            f"({self.busy_fraction():.2f} busy-worker-equivalents), {self.new_tokens} new "
            f"token(s) ({self.tokens_per_second():.0f} tok/s); resolved {self.resolved}, "
            f"unresolved {self.unresolved}"
            + (f"; model load excluded [{loads}]" if loads else "")
        )


def grounded_worker_preflight(args, workers):
    """Fail loudly BEFORE loading anything if N replicas cannot be placed.

    The api backend is stateless HTTP, so it needs no local devices; the transformers
    backend needs one visible GPU per worker: each spawned child pins itself to exactly
    one of them via CUDA_VISIBLE_DEVICES.
    """
    if workers < 1:
        raise RuntimeError(f"--generate-grounded-workers must be >= 1, got {workers}")
    if workers == 1 or args.teacher_backend != "transformers":
        return
    # Prefer the scheduler's own list: it is authoritative, it is what each child will be
    # pinned to, and reading it avoids importing torch (and touching CUDA) in the parent
    # that is about to spawn.
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip():
        available = len([part for part in str(raw).split(",") if part.strip()])
    else:
        import torch

        available = int(torch.cuda.device_count())
    if workers > available:
        raise RuntimeError(
            f"--generate-grounded-workers {workers} exceeds the {available} visible CUDA "
            "device(s): each worker owns one teacher replica pinned to its own GPU "
            "(pinned via CUDA_VISIBLE_DEVICES). Lower the worker count or expose more "
            "GPUs (check CUDA_VISIBLE_DEVICES)."
        )


def _grounded_paths(split, args):
    cfg = mode_config(args)
    if split == "train":
        return (cfg["train_jsonl"], cfg["grounded_train_jsonl"],
                f"{cfg['split_name_prefix']}-train")
    if split == "eval":
        return (cfg["eval_jsonl"], cfg["grounded_eval_jsonl"],
                f"{cfg['split_name_prefix']}-eval")
    raise ValueError(f"unknown split {split!r}")


def group_grounded_rows_by_item(base_rows):
    """(ordered dataset_index list, {dataset_index: {orientation: row}}). Both
    orientations of an item always live in the same split, so every group is complete."""
    order, by_index = [], {}
    for row in base_rows:
        idx = row["dataset_index"]
        if idx not in by_index:
            order.append(idx)
            by_index[idx] = {}
        by_index[idx][row["orientation"]] = row
    for idx in order:
        missing = set(ORIENTATIONS) - set(by_index[idx])
        if missing:
            raise RuntimeError(f"item {idx} is missing orientation(s) {sorted(missing)}")
    return order, by_index


class GroundedRecordCorrupt(ValueError):
    """The record is structurally broken, mis-bound, or tampered with.

    ALWAYS fatal: the loader refuses the artifact and a resume refuses to continue. Covers
    schema/key-set, content-hash, identity-field, base-row-binding, orientation, status
    shape and re-render/audit/analysis-hash failures.
    """


class GroundedRecordStale(ValueError):
    """The record is internally sound but does not match requested provenance.

    Prompt-version or audit-threshold drift, another teacher model/backend, or a teacher
    prompt digest that cannot be re-derived makes the row regenerable during resume and
    invalid for direct loading.
    """


def _assert_grounded_failures_shape(record):
    """`failures` must be a list of dicts whose 'epoch' (when present) is int-coercible.

    grounded_next_epoch() reads this list, and it is read for STALE records too — so the
    shape is validated before any staleness raise, and a violation is CORRUPT (fatal),
    never regenerable. Without this a hand-edited `failures` would surface as a bare
    AttributeError/TypeError instead of the promised precise error.
    """
    failures = record.get("failures")
    if not isinstance(failures, list):
        raise GroundedRecordCorrupt(
            f"failures must be a list, got {type(failures).__name__}")
    for index, failure in enumerate(failures):
        if not isinstance(failure, dict):
            raise GroundedRecordCorrupt(
                f"failures[{index}] must be an object, got {type(failure).__name__}")
        epoch = failure.get("epoch", 0)
        if isinstance(epoch, bool):
            raise GroundedRecordCorrupt(f"failures[{index}].epoch is a bool, not an int")
        try:
            int(epoch)
        except (TypeError, ValueError) as exc:
            raise GroundedRecordCorrupt(
                f"failures[{index}].epoch {epoch!r} is not int-coercible") from exc


def _verify_grounded_record(record, item, rows_by_orientation, split_name, story, cfg, mode):
    """Raise ValueError unless `record` is a valid, bound, guardrail-passing grounded
    record for the freshly rebuilt rows. Re-runs EVERY audit; re-renders the analysis;
    re-derives the teacher prompt hash. The artifact is trusted for nothing.

    Raises ``GroundedRecordCorrupt`` for fatal content defects or
    ``GroundedRecordStale`` when resume may regenerate the row.
    """
    try:
        grounded_content_hash(record)  # exact key set (raises with a precise message)
    except ValueError as exc:
        raise GroundedRecordCorrupt(str(exc)) from exc
    # BEFORE any stale check: grounded_next_epoch() reads `failures` on stale records too,
    # so a malformed list must fail as CORRUPT here rather than escaping later as a bare
    # AttributeError/TypeError from the epoch read.
    _assert_grounded_failures_shape(record)
    if record.get("grounded_prompt_version") != GROUNDED_PROMPT_VERSION:
        raise GroundedRecordStale(
            f"grounded_prompt_version {record.get('grounded_prompt_version')!r} != "
            f"{GROUNDED_PROMPT_VERSION!r}; regenerate this stale record"
        )
    if record.get("audit_thresholds") != grounded_audit_thresholds():
        raise GroundedRecordStale("audit_thresholds do not match the current code")
    for field, expected in (("record_id", item["pair_id"]), ("pair_id", item["pair_id"]),
                            ("dataset_index", item["dataset_index"]),
                            ("story_title", item["story_title"]),
                            ("split", split_name), ("dataset", cfg["dataset"]),
                            ("mode", mode),
                            ("source_dataset_index", item.get("source_dataset_index"))):
        if record.get(field) != expected:
            raise GroundedRecordCorrupt(
                f"{field} {record.get(field)!r} != rebuilt {expected!r}")
    expected_base = {o: rows_by_orientation[o]["row_sha256"] for o in ORIENTATIONS}
    if record.get("base_row_sha256") != expected_base:
        raise GroundedRecordCorrupt(
            "base_row_sha256 does not match the rebuilt rows (prompt/target drift)")
    orientation = record.get("teacher_orientation")
    if orientation != grounded_teacher_orientation(item["pair_id"]):
        raise GroundedRecordCorrupt(
            f"teacher_orientation {orientation!r} is not the hashed canonical one")
    canonical_row = rows_by_orientation[orientation]
    if record.get("teacher_row_id") != canonical_row["row_id"]:
        raise GroundedRecordCorrupt(
            "teacher_row_id does not match the canonical orientation row")
    recomputed = grounded_content_hash(record)
    if record.get("grounded_content_sha256") != recomputed:
        raise GroundedRecordCorrupt(
            "grounded_content_sha256 mismatch — the record was edited after it was written"
        )
    status = record.get("status")
    if status not in ("resolved", "unresolved"):
        raise GroundedRecordCorrupt(f"unknown status {status!r}")
    if status == "unresolved":
        for field in ("evidence", "check_text", "winner_debater", "analysis",
                      "analysis_chars", "audit", "teacher_prompt_sha256", "teacher_tier",
                      "teacher_attempt", "teacher_temperature", "grounded_sha256",
                      "grounded_row_sha256"):
            if record.get(field) is not None:
                raise GroundedRecordCorrupt(
                    f"unresolved record carries a non-null {field}")
        if not record.get("failures"):
            raise GroundedRecordCorrupt("unresolved record carries no failure reasons")
        return None

    analysis = record.get("analysis")
    if not isinstance(analysis, str):
        raise GroundedRecordCorrupt("missing/non-string analysis")
    try:
        assert_rendered_analysis_shape(analysis)
        parsed = {
            "evidence": {d: record["evidence"][d]["span"] for d in GROUNDED_DEBATERS},
            "check_text": record["check_text"],
            "winner_debater": record["winner_debater"],
        }
        evidence, check_text, winner, rendered, audit = validate_grounded(
            parsed, item, canonical_row, story)
    except (ValueError, KeyError, TypeError) as exc:
        raise GroundedRecordCorrupt(str(exc)) from exc
    if rendered != analysis:
        raise GroundedRecordCorrupt(
            "stored analysis does not re-render from its own audited fields")
    if record.get("evidence") != evidence:
        raise GroundedRecordCorrupt(
            "stored evidence does not match the recomputed span audit")
    if record.get("audit") != audit:
        raise GroundedRecordCorrupt("stored audit does not match the recomputed audit")
    if record.get("check_text") != check_text or record.get("winner_debater") != winner:
        raise GroundedRecordCorrupt(
            "stored check_text/winner do not survive re-validation")
    if record.get("analysis_chars") != len(analysis):
        raise GroundedRecordCorrupt("analysis_chars does not match the analysis")
    if record.get("grounded_sha256") != grounded_text_hash(analysis):
        raise GroundedRecordCorrupt("grounded_sha256 mismatch")
    expected_row_hash = grounded_row_hash(
        canonical_row["messages"], canonical_row["target"], analysis)
    if record.get("grounded_row_sha256") != expected_row_hash:
        raise GroundedRecordCorrupt("grounded_row_sha256 mismatch")
    saw_gold = bool(record.get("teacher_saw_gold"))
    if record.get("teacher_tier") != ("gold" if saw_gold else "blind"):
        raise GroundedRecordCorrupt("teacher_tier disagrees with teacher_saw_gold")
    teacher_messages = build_grounded_teacher_messages(
        canonical_row, saw_gold, _gold_answer_text(item), _gold_debater_for_row(canonical_row))
    if teacher_prompt_hash(teacher_messages) != record.get("teacher_prompt_sha256"):
        raise GroundedRecordStale(
            "teacher_prompt_sha256 mismatch (teacher prompt drift, or built from the wrong "
            "inputs); regenerate this record"
        )
    return analysis


def _write_grounded_snapshot(path, records):
    """Durably replace `path` with `records` — the SPARSE SNAPSHOT primitive.

    Called after every accepted completion with every valid record known so far (reused +
    newly accepted), already sorted by canonical slot. Whole-file atomic replace rather
    than append: an appended JSONL can be left with a torn final line by a crash, while
    atomic replacement ensures a reader sees either the prior complete snapshot or the
    replacement.

    Durability chain: write tmp -> flush -> fsync(tmp) -> os.replace -> fsync(parent dir).
    The directory fsync is best-effort (not every platform/filesystem supports it).
    Deliberately NOT write_jsonl_atomic: that one is shared with the rationale/Q_H arms and
    stays byte-for-byte as it is.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _read_grounded_partial(path, order, items, split_name, story_by_index, cfg, args):
    """STRICT read of a (possibly sparse) grounded artifact for resume.

    Accepts a well-formed sparse SUBSET in canonical relative order. Fails loud — never
    last-wins, never silently skips or drops — on a malformed line, a duplicate or foreign
    record_id, a record whose dataset_index contradicts its record_id, records out of
    canonical relative order, or any CORRUPT-class verification failure.

    Returns {dataset_index: ("reuse", record) | ("pending", record-or-None)}:
      * resolved + fully verified + same teacher config  -> reuse;
      * GroundedRecordStale, or a different teacher model/backend               -> pending;
      * verified unresolved                                                     -> pending
        (its epoch is read by the caller only AFTER this audit, never before).
    """
    if not os.path.exists(path):
        return {}, {"records": 0, "reuse": 0, "stale": 0, "unresolved": 0}
    id_to_index = {items[idx]["pair_id"]: idx for idx in order}
    slot_of_index = {idx: slot for slot, idx in enumerate(order)}
    outcomes, seen_ids, previous_slot = {}, set(), -1
    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()
    stats = {"records": 0, "reuse": 0, "stale": 0, "unresolved": 0}
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise RuntimeError(
                f"{path}:{lineno}: blank line in the grounded artifact. The atomic "
                "snapshot writer never emits one, so this file was edited or truncated; "
                "refusing to resume from it ('never silently skip')."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{path}:{lineno}: malformed JSON in the grounded artifact ({exc}). This "
                "file is written by whole-file atomic replace, so a torn or hand-edited "
                "line means corruption; refusing to resume from it."
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"{path}:{lineno}: record is not a JSON object")
        record_id = record.get("record_id")
        if record_id not in id_to_index:
            raise RuntimeError(
                f"{path}:{lineno}: record_id {record_id!r} does not belong to this split "
                f"({split_name}); refusing to resume from a foreign or stale artifact."
            )
        if record_id in seen_ids:
            raise RuntimeError(
                f"{path}:{lineno}: duplicate record_id {record_id!r}; the snapshot writer "
                "emits each item exactly once."
            )
        seen_ids.add(record_id)
        dataset_index = id_to_index[record_id]
        if record.get("dataset_index") != dataset_index:
            raise RuntimeError(
                f"{path}:{lineno}: record {record_id!r} claims dataset_index "
                f"{record.get('dataset_index')!r} but that record_id maps to "
                f"{dataset_index}; refusing a mismatched binding."
            )
        slot = slot_of_index[dataset_index]
        if slot <= previous_slot:
            raise RuntimeError(
                f"{path}:{lineno}: record {record_id!r} is out of canonical order (slot "
                f"{slot} follows slot {previous_slot}); a snapshot is always slot-sorted."
            )
        previous_slot = slot
        stats["records"] += 1

        item = items[dataset_index]
        try:
            _verify_grounded_record(record, item, story_by_index[dataset_index]["rows"],
                                    split_name, story_by_index[dataset_index]["story"],
                                    cfg, args.mode)
        except GroundedRecordStale:
            outcomes[dataset_index] = ("pending", record)
            stats["stale"] += 1
            continue
        except GroundedRecordCorrupt as exc:
            raise RuntimeError(
                f"{path}:{lineno}: record {record_id!r} is corrupt: {exc}"
            ) from exc
        if record.get("status") != "resolved":
            # A verified unresolved row is pending; read its retry epoch only after audit.
            outcomes[dataset_index] = ("pending", record)
            stats["unresolved"] += 1
            continue
        if (record.get("teacher_model") != args.teacher_model
                or record.get("teacher_backend") != args.teacher_backend):
            outcomes[dataset_index] = ("pending", record)
            stats["stale"] += 1
            continue
        outcomes[dataset_index] = ("reuse", record)
        stats["reuse"] += 1
    return outcomes, stats


def _grounded_manifest(records, split, split_name, args, cfg, base_jsonl):
    resolved = [r for r in records if r["status"] == "resolved"]
    unresolved = [r for r in records if r["status"] != "resolved"]
    gold = [r for r in resolved if r["teacher_saw_gold"]]
    lengths = sorted(r["analysis_chars"] for r in resolved) or [0]
    verified_counts = {
        d: sum(1 for r in resolved if r["evidence"][d]["verified"]) for d in GROUNDED_DEBATERS
    }
    winner_counts = Counter(r["winner_debater"] for r in resolved)
    return {
        "dataset": cfg["dataset"],
        "mode": args.mode,
        "split": split,
        "split_name": split_name,
        "grounded_prompt_version": GROUNDED_PROMPT_VERSION,
        "item_count": len(records),
        "resolved_count": len(resolved),
        "blind_count": len(resolved) - len(gold),
        "gold_count": len(gold),
        "gold_tier_fraction": (len(gold) / len(resolved)) if resolved else 0.0,
        "unresolved_count": len(unresolved),
        "unresolved_pair_ids": [r["pair_id"] for r in unresolved],
        "span_verified_counts": verified_counts,
        "winner_side_counts": dict(winner_counts),
        "analysis_chars": {
            "min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1],
        },
        "combined_sha256": grounded_combined_hash(records),
        "base_jsonl": base_jsonl,
        "teacher_model": args.teacher_model,
        "teacher_backend": args.teacher_backend,
        "teacher_max_new_tokens": args.teacher_max_new_tokens,
        # Transport provenance. MANIFEST ONLY: no record field is added, so
        # GROUNDED_RECORD_FIELDS / grounded_content_hash / combined_sha256 are untouched.
        "teacher_base_url": (args.teacher_base_url
                             if args.teacher_backend == "api" else None),
        "generation_concurrency": grounded_concurrency_of(args),
        "teacher_api": ({"timeout": GROUNDED_API_TIMEOUT,
                         "transport_retries": GROUNDED_API_TRANSPORT_RETRIES,
                         "seeded_requests": True}
                        if args.teacher_backend == "api" else None),
        "greedy_temperature": TEACHER_GREEDY_TEMPERATURE,
        "retry_temperature": TEACHER_RETRY_TEMPERATURE,
        "blind_attempts": GROUNDED_BLIND_ATTEMPTS,
        "gold_attempts": GROUNDED_GOLD_ATTEMPTS,
        "top_p": TEACHER_TOP_P,
        "audit_thresholds": grounded_audit_thresholds(),
        "teacher_output_schema": GROUNDED_TEACHER_OUTPUT_SCHEMA,
        "analysis_schema": GROUNDED_ANALYSIS_SCHEMA,
        "h_hygiene": GROUNDED_H_HYGIENE_NOTE,
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _verify_grounded_manifest(manifest_path, records, split):
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"{manifest_path} not found — (re)generate with "
                           f"--generate-grounded --split {split}.")
    with open(manifest_path) as f:
        man = json.load(f)
    if man.get("grounded_prompt_version") != GROUNDED_PROMPT_VERSION:
        raise RuntimeError(
            f"{manifest_path}: grounded_prompt_version "
            f"{man.get('grounded_prompt_version')!r} != {GROUNDED_PROMPT_VERSION!r}; "
            "regenerate this stale artifact"
        )
    if man.get("audit_thresholds") != grounded_audit_thresholds():
        raise RuntimeError(f"{manifest_path}: audit thresholds do not match the current code")
    if man.get("split") != split:
        raise RuntimeError(f"manifest split {man.get('split')!r} != {split!r}")
    if man.get("item_count") != len(records):
        raise RuntimeError(f"manifest item_count {man.get('item_count')} != {len(records)}")
    expected = grounded_combined_hash(records)
    if man.get("combined_sha256") != expected:
        raise RuntimeError(f"{manifest_path}: combined_sha256 mismatch — partial / reordered "
                           "/ swapped grounded artifact vs manifest.")
    return man


def load_and_verify_grounded_records(base_rows, items, stories_by_title, split, args):
    """Load the grounded artifact, verify order + key sets + hashes + manifest + every
    audit against the rebuilt base rows, apply the unresolved policy, and return
    {dataset_index: analysis-or-None}. Raises (never regenerates)."""
    cfg = mode_config(args)
    base_jsonl, grounded_path, split_name = _grounded_paths(split, args)
    manifest_path = cfg["grounded_manifest_template"].format(split=split)
    if not os.path.exists(grounded_path):
        raise RuntimeError(
            f"{grounded_path} not found. Grounded targets are generated offline — run "
            f"`--generate-grounded --split {split}` (cluster/GPU) before training. "
            "Training never regenerates them online.")
    with open(grounded_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    order, rows_by_index = group_grounded_rows_by_item(base_rows)
    if len(records) != len(order):
        raise RuntimeError(
            f"{grounded_path}: {len(records)} records != {len(order)} split items — an "
            "item was dropped or added; the artifact must carry exactly one record per item"
        )
    _verify_grounded_manifest(manifest_path, records, split)

    analyses, unresolved, tiers = {}, [], Counter()
    for record, dataset_index in zip(records, order):
        item = items[dataset_index]
        if record.get("dataset_index") != dataset_index:
            raise RuntimeError(
                f"{grounded_path}: record order mismatch at {record.get('record_id')!r} "
                f"(dataset_index {record.get('dataset_index')!r} != {dataset_index})"
            )
        story = stories_by_title[item["story_title"]]
        try:
            analysis = _verify_grounded_record(
                record, item, rows_by_index[dataset_index], split_name, story, cfg, args.mode)
        except ValueError as exc:
            raise RuntimeError(f"{grounded_path}: record {record.get('record_id')!r}: {exc}") from exc
        analyses[dataset_index] = analysis
        if analysis is None:
            unresolved.append(record["pair_id"])
        else:
            tiers[record["teacher_tier"]] += 1

    policy = grounded_unresolved_of(args)
    if unresolved and policy == GROUNDED_UNRESOLVED_FAIL:
        raise RuntimeError(
            f"{grounded_path}: {len(unresolved)} item(s) were never resolved by the "
            f"teacher (e.g. {unresolved[:5]}). The artifact and its per-attempt failure "
            f"reasons are on disk — inspect {manifest_path}, then either re-run "
            f"`--generate-grounded --split {split}` (unresolved records are always "
            "retried, at a fresh sampling epoch) or re-run this chain with "
            "`--grounded-unresolved answer-only` to "
            "train those items on the direct forced-choice view only. No item is dropped "
            "under either policy."
        )
    resolved_n = len(analyses) - len(unresolved)
    gold_fraction = (tiers["gold"] / resolved_n) if resolved_n else 0.0
    print(f"Grounded verify [{split}]: {resolved_n}/{len(analyses)} items resolved "
          f"(blind {tiers['blind']}, gold {tiers['gold']}, gold fraction {gold_fraction:.2f}); "
          f"unresolved {len(unresolved)} (policy {policy}).")
    if gold_fraction > GROUNDED_GOLD_TIER_WARN_FRACTION:
        print(f"WARNING: gold-tier fraction {gold_fraction:.2f} exceeds "
              f"{GROUNDED_GOLD_TIER_WARN_FRACTION:.2f} — the grounding teacher is mostly "
              "rationalising from the gold note; interpret M2 results split by tier.",
              file=sys.stderr)
    return analyses


def generate_grounded(args):
    """--generate-grounded --split {train,eval}: run the tier ladder per item, audit,
    render, hash, write the artifact + manifest, then re-verify with the training-time
    loader.

    ALWAYS writes every record, including unresolved ones with their per-attempt failure
    reasons: the unresolved POLICY is applied by the loader, not here, so a fail-closed
    run still leaves a fully inspectable artifact on disk.

    Resume: a RESOLVED record is reused only if its teacher model/backend and prompt
    version match this run and it fully re-verifies. An UNRESOLVED record is never a
    cached result — it is retried at the next RETRY EPOCH (grounded_next_epoch), which
    salts the sampled-attempt seeds so the retry draws differently instead of replaying
    the same failures; the greedy attempts stay deterministic by design.
    """
    split = args.split
    cfg = mode_config(args)
    items, base_rows, base_jsonl, split_name = _load_grounded_split(split, args)
    stories_by_title = load_stories_for_items(items, cfg)
    _, grounded_path, _ = _grounded_paths(split, args)
    manifest_path = cfg["grounded_manifest_template"].format(split=split)
    order, rows_by_index = group_grounded_rows_by_item(base_rows)

    from tqdm import tqdm  # lazy: only the GPU generate path needs the progress bar

    workers = grounded_workers_of(args)
    concurrency = grounded_concurrency_of(args)
    grounded_worker_preflight(args, workers)

    # PASS 1 (parent, no GPU): STRICT resume + reuse decisions + retry epochs. Every item
    # lands in a fixed SLOT, so the snapshot is canonically ordered whatever order the
    # workers finish in.
    story_by_index = {idx: {"rows": rows_by_index[idx],
                            "story": stories_by_title[items[idx]["story_title"]]}
                      for idx in order}
    prior_outcomes, resume_stats = _read_grounded_partial(
        grounded_path, order, items, split_name, story_by_index, cfg, args)
    if resume_stats["records"]:
        print(f"Resume [{split}]: {resume_stats['records']} prior record(s) — "
              f"{resume_stats['reuse']} reusable, {resume_stats['stale']} provenance-stale, "
              f"{resume_stats['unresolved']} unresolved; all audited before use.")

    results = [None] * len(order)
    settled = set()      # slots whose FINAL record for this run is in `results`
    carried = set()      # slots still holding a PRIOR record, pending regeneration
    pending = []
    reused = regenerated = retried = still_unresolved = 0
    for slot, dataset_index in enumerate(order):
        item = items[dataset_index]
        rows = rows_by_index[dataset_index]
        canonical_row = rows[grounded_teacher_orientation(item["pair_id"])]
        story = stories_by_title[item["story_title"]]
        outcome, existing = prior_outcomes.get(dataset_index, (None, None))
        if outcome == "reuse":
            results[slot] = existing
            settled.add(slot)
            reused += 1
            continue
        # A record of FAILURE is never a cached result: always retry. The epoch salts the
        # sampled-attempt seeds so the retry genuinely re-draws instead of replaying the
        # same four failures, which is what makes "always retried" a real guarantee. The
        # record was fully audited above, so its epoch is only read AFTER verification.
        epoch = grounded_next_epoch(existing)
        if existing is not None:
            # CARRY the prior record in every snapshot until _accept replaces it. Without
            # this, a crash mid-retry would leave the artifact with NO record for the item,
            # and the next resume would compute epoch 0 and replay the very seeds that
            # already failed — silently losing the retry-epoch progression.
            results[slot] = existing
            carried.add(slot)
            if existing.get("status") != "resolved":
                retried += 1
        pending.append({"slot": slot, "item": item, "rows": rows,
                        "canonical_row": canonical_row, "story": story, "epoch": epoch})

    progress = _GroundedProgress(len(pending), reused)

    # AFTER the resume pass, and only when this run will actually issue requests: a fully
    # resolved artifact (the common "did it finish?" re-run, and the re-verify pass below)
    # must not require a live server. Independent of concurrency — a serial API run probes
    # too, because a wrong port or model id is exactly as fatal at concurrency 1.
    if pending and args.teacher_backend == "api":
        served = _probe_grounded_endpoint(args.teacher_base_url, args.teacher_model)
        print(f"Teacher endpoint {args.teacher_base_url} ready; serves {served} "
              f"(using {args.teacher_model!r}, concurrency {concurrency}).")

    def _snapshot():
        """Durably persist every valid record known so far, slot-sorted."""
        _write_grounded_snapshot(grounded_path,
                                 [r for r in results if r is not None])

    if pending and any(record is not None for record in results):
        # Reused AND carried-over records are durable before any generation starts.
        _snapshot()

    pbar = tqdm(total=len(order), desc=f"generate-grounded[{split}]", unit="item")
    pbar.update(reused)

    def _accept(task, record, resolved, telemetry=None):
        """Verify -> place -> PERSIST. Returns only after the snapshot is on disk, which
        is what bounds the crash window to one unpersisted item per child."""
        nonlocal regenerated, still_unresolved
        slot = task["slot"]
        item = task["item"]
        try:
            _verify_grounded_record(record, item, task["rows"], split_name, task["story"],
                                    cfg, args.mode)
        except ValueError as exc:
            raise RuntimeError(
                f"grounded worker returned a record for {item['pair_id']!r} that fails "
                f"verification: {exc}"
            ) from exc
        results[slot] = record          # replaces any carried-over prior record
        settled.add(slot)
        carried.discard(slot)
        if resolved:
            regenerated += 1
        else:
            still_unresolved += 1
            failures = record["failures"]
            pbar.write(f"  {item['pair_id']}: UNRESOLVED after "
                       f"{GROUNDED_MAX_ATTEMPTS} attempts; last failure: "
                       f"{failures[-1]['reason'] if failures else 'n/a'}")
        _snapshot()
        progress.record_item(record, telemetry)
        pbar.update(1)
        pbar.set_postfix(reused=reused, retried=retried, **progress.postfix())

    try:
        if pending and concurrency > 1:
            # API backend only (main() rejects every other combination): one shared server,
            # N bounded in-flight HTTP requests, still one writer on this thread.
            _run_grounded_thread_pool(
                args, cfg, split_name, concurrency, pending, _accept, progress)
        elif pending and workers == 1:
            # One local replica with no child queue; this also handles serial API
            # generation when concurrency is one.
            stats = {"calls": 0, "new_tokens": 0, "seconds": 0.0}
            generate_fn = _load_grounded_teacher(args, stats=stats)
            progress.start()
            for task in pending:
                before = dict(stats)
                record, resolved = _build_grounded_record_for_task(
                    task, generate_fn, args, cfg, split_name)
                _accept(task, record, resolved,
                        {k: stats[k] - before.get(k, 0)
                         for k in ("calls", "new_tokens", "seconds")})
        elif pending:
            _run_grounded_worker_pool(
                args, cfg, split, split_name, workers, pending, _accept, progress, pbar)
    finally:
        pbar.close()

    # A carried-over prior record is NOT a completed one: completion is `settled`, so a
    # never-regenerated item is still a hole even though its slot is non-empty.
    missing = [order[i] for i in range(len(order)) if i not in settled]
    if missing:
        raise RuntimeError(
            f"grounded generation produced no record for dataset index(es) {missing[:5]}; "
            "refusing to write an artifact with holes."
        )
    if carried:
        raise RuntimeError(
            f"{len(carried)} carried-over prior record(s) were never replaced; refusing to "
            "canonicalise an artifact that still contains pre-run records."
        )
    out_records = results
    if pending:
        print(progress.summary())
    # Canonicalize the complete slot-ordered set with the durable snapshot primitive.
    _write_grounded_snapshot(grounded_path, out_records)
    write_json_atomic(manifest_path,
                      _grounded_manifest(out_records, split, split_name, args, cfg, base_jsonl))
    print(f"Wrote {grounded_path} ({len(out_records)} records; reused {reused}, "
          f"generated {regenerated}, retried-unresolved {retried}, still-unresolved "
          f"{still_unresolved}; {workers} teacher worker(s); concurrency {concurrency}) "
          f"+ {manifest_path}.")

    # Re-verify with the EXACT training-time loader, under a policy that never refuses:
    # generation reports, the loader's policy decides downstream.
    requested_policy = grounded_unresolved_of(args) or GROUNDED_UNRESOLVED_DEFAULT
    verify_args = copy.copy(args)
    verify_args.grounded_unresolved = GROUNDED_UNRESOLVED_ANSWER_ONLY
    load_and_verify_grounded_records(base_rows, items, stories_by_title, split, verify_args)
    print(f"--generate-grounded [{split}] complete and re-verified. NOTE: this "
          f"generation-time re-verification ran with unresolved policy "
          f"{GROUNDED_UNRESOLVED_ANSWER_ONLY!r} (inspection only, so the artifact is "
          f"always written and readable); your requested policy {requested_policy!r} is "
          "enforced by --check-grounded / --check-tokenizer / training.")


def _grounded_child_payload(args, cfg, split_name, device_index,
                            parent_visible=_GROUNDED_VISIBLE_UNSET):
    """Everything ONE child needs, as plain picklable data (no Namespace-only types).

    `parent_visible` is snapshotted ONCE by the caller before any child starts: each child
    rewrites CUDA_VISIBLE_DEVICES in its own process, and the parent's mapping must be
    computed from the launch-time list, never from an environment in flux.
    """
    return {
        "device_index": device_index,
        "visible_device": grounded_child_visible_device(device_index, parent_visible),
        "args": dict(vars(args)),
        "cfg": cfg,
        "split_name": split_name,
    }


def _run_grounded_worker_pool(args, cfg, split, split_name, workers, pending, accept,
                              progress, pbar):
    """Spawn one isolated child per GPU and drive them with credit-based backpressure.

    Invariants:
      * ALL-READY BARRIER — not one task is dispatched until every child reports ready;
      * AT MOST ONE unacknowledged task per child — the next task goes out only after the
        parent has verified AND durably persisted the previous result, which is what makes
        the crash window one unpersisted item per child;
      * EXACT-ONCE — a slot leaves `pending` once and enters `done` once; an unexpected or
        duplicate result slot fails loud;
      * NO ORPHANS — sentinels, join, terminate, kill, join again, always in `finally`.
    """
    ctx = grounded_mp_context()
    result_q = ctx.Queue()
    task_qs, children = [], []
    queue_of_pending = list(pending)
    by_slot = {task["slot"]: task for task in pending}
    inflight, done = {}, set()
    ready = set()

    def _alive():
        return [c for c in children if c.is_alive()]

    def _drain_result(deadline_message):
        """One result, or a loud failure. Never blocks forever: on every poll timeout it
        re-checks liveness, so a dead child ends the wait instead of deadlocking."""
        while True:
            try:
                return result_q.get(timeout=GROUNDED_RESULT_POLL_SECONDS)
            except Exception:  # queue.Empty (the fake context may raise its own)
                pass
            if len(_alive()) < len(children):
                dead = [c.name for c in children if not c.is_alive()]
                raise RuntimeError(
                    f"{deadline_message}: worker process(es) {dead} exited unexpectedly "
                    f"(exit codes {[c.exitcode for c in children if not c.is_alive()]}). "
                    + _grounded_inflight_note(inflight, by_slot)
                )

    try:
        print(f"Spawning {workers} isolated teacher process(es) (one per GPU) for "
              f"generate-grounded[{split}] ...")
        # Snapshot the launch-time device list once: every child rewrites
        # CUDA_VISIBLE_DEVICES in its own process, so re-reading it per child would map
        # later workers against an environment that is already being rewritten.
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        for device_index in range(workers):
            task_q = ctx.Queue()
            payload = _grounded_child_payload(args, cfg, split_name, device_index,
                                              parent_visible)
            child = ctx.Process(target=_grounded_child_main,
                                args=(payload, task_q, result_q),
                                name=f"grounded-teacher-{device_index}", daemon=True)
            child.start()
            task_qs.append(task_q)
            children.append(child)

        # ---- all-ready barrier ------------------------------------------------
        barrier_deadline = time.monotonic() + GROUNDED_CHILD_READY_TIMEOUT
        while len(ready) < workers:
            if time.monotonic() > barrier_deadline:
                raise RuntimeError(
                    f"only {len(ready)}/{workers} teacher worker(s) became ready within "
                    f"{GROUNDED_CHILD_READY_TIMEOUT:.0f}s; aborting before dispatch."
                )
            message = _drain_result("worker startup")
            kind = message.get("kind")
            if kind == "fatal":
                raise RuntimeError(
                    f"teacher worker {message.get('device_index')} failed to start:\n"
                    f"{message.get('traceback')}"
                )
            if kind != "ready":
                raise RuntimeError(f"unexpected pre-ready message from a worker: {kind!r}")
            ready.add(message["device_index"])
            progress.record_ready(message["device_index"], message.get("load_seconds", 0.0))
            pbar.write(f"  worker {message['device_index']} ready "
                       f"({message.get('load_seconds', 0.0):.0f}s model load)")

        # ---- credit-based dispatch: one outstanding task per child ------------
        progress.start()   # generation clock: after load, before the first dispatch

        def _dispatch(device_index):
            if not queue_of_pending:
                return
            task = queue_of_pending.pop(0)
            inflight[device_index] = task["slot"]
            task_qs[device_index].put({
                "slot": task["slot"], "item": task["item"], "rows": task["rows"],
                "canonical_row": task["canonical_row"], "story": task["story"],
                "epoch": task["epoch"],
            })

        for device_index in range(workers):
            _dispatch(device_index)

        while inflight:
            message = _drain_result("waiting for a worker result")
            kind = message.get("kind")
            device_index = message.get("device_index")
            if kind == "error":
                raise RuntimeError(
                    f"teacher worker {device_index} raised while building slot "
                    f"{message.get('slot')}:\n{message.get('traceback')}"
                )
            if kind != "record":
                raise RuntimeError(f"unexpected message from worker {device_index}: {kind!r}")
            slot = message["slot"]
            expected = inflight.get(device_index)
            if expected is None or slot != expected:
                raise RuntimeError(
                    f"unexpected result slot {slot} from worker {device_index} "
                    f"(expected {expected!r}); refusing to accept an unrequested record."
                )
            if slot in done:
                raise RuntimeError(
                    f"duplicate result for slot {slot} from worker {device_index}; "
                    "exact-once accounting violated."
                )
            del inflight[device_index]
            # Verify + persist BEFORE handing this worker its next task.
            accept(by_slot[slot], message["record"], message.get("resolved", False),
                   message.get("telemetry"))
            done.add(slot)
            _dispatch(device_index)
    finally:
        for task_q in task_qs:
            try:
                task_q.put(None)
            except Exception:  # noqa: BLE001 - a dead child's queue may already be broken
                pass
        for child in children:
            try:
                child.join(GROUNDED_CHILD_JOIN_TIMEOUT)
            except Exception:  # noqa: BLE001
                pass
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            if child.is_alive():
                try:
                    child.join(GROUNDED_CHILD_JOIN_TIMEOUT)
                    if child.is_alive():
                        child.kill()
                        child.join(GROUNDED_CHILD_JOIN_TIMEOUT)
                except Exception:  # noqa: BLE001
                    pass


def _run_grounded_thread_pool(args, cfg, split_name, concurrency, pending, accept,
                              progress):
    """API-backend sibling of _run_grounded_worker_pool: N bounded in-flight HTTP requests
    against ONE shared vLLM server, instead of N GPU-pinned processes.

    Invariants kept from the process pool:
      * SINGLE WRITER — `accept` (verify -> place -> fsync snapshot) runs only on this
        thread; worker threads build records and nothing else;
      * CANONICAL ORDER — every result lands in its fixed slot, so completion order cannot
        reorder the artifact; finished futures are drained in slot order purely so failure
        messages and progress output are stable;
      * EXACT-ONCE — a task is submitted once and accepted once;
      * NO ORPHANS — on any failure every queued future is cancelled and the executor still
        joins the in-flight requests (each request has GROUNDED_API_TIMEOUT, with bounded
        transport retries inside the request).

    Deliberately RELAXED, and the only invariant that changes: the crash window is at most
    `concurrency` unpersisted in-flight items, not one per worker. A resume redoes exactly
    those; every accepted record is already fsynced.

    Thread safety: each thread lazily builds its OWN teacher closure (its own HTTP client
    and its own telemetry dict), so the per-item before/after delta is exact and no sampling
    state is shared. _GroundedApiClient additionally takes temperature/seed per call, so
    there is no mutable lm dict to race on even if a client were ever shared.
    """
    import concurrent.futures as cf
    import threading

    local = threading.local()

    def _teacher():
        if not hasattr(local, "generate"):
            local.stats = {"calls": 0, "new_tokens": 0, "seconds": 0.0}
            local.generate = _load_grounded_teacher(args, stats=local.stats)
        return local.generate, local.stats

    def _work(task):
        generate_fn, stats = _teacher()
        before = dict(stats)
        record, resolved = _build_grounded_record_for_task(
            task, generate_fn, args, cfg, split_name)
        return record, resolved, {k: stats[k] - before.get(k, 0)
                                  for k in ("calls", "new_tokens", "seconds")}

    queue_of_pending = list(pending)
    inflight = {}   # future -> task
    print(f"Dispatching {len(queue_of_pending)} item(s) over {concurrency} concurrent "
          f"teacher request(s) to {args.teacher_base_url} ...")
    # Thread prefix is deliberately NOT 'grounded-teacher': that name is parsed by the
    # process-pool test fake to identify a replica.
    with cf.ThreadPoolExecutor(max_workers=concurrency,
                               thread_name_prefix="grounded-http") as pool:
        try:
            progress.start()   # generation clock: before the first dispatch
            while queue_of_pending and len(inflight) < concurrency:
                task = queue_of_pending.pop(0)
                inflight[pool.submit(_work, task)] = task
            while inflight:
                done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: inflight[f]["slot"]):
                    # Keep the task in `inflight` until both the worker result and the
                    # main-thread verify/fsync accept succeed. If either raises, the
                    # failure diagnostic must name this unpersisted item so resume is
                    # actionable; removing it before future.result() would lose it.
                    task = inflight[future]
                    record, resolved, telemetry = future.result()   # re-raises loudly
                    accept(task, record, resolved, telemetry)
                    del inflight[future]
                    if queue_of_pending:
                        nxt = queue_of_pending.pop(0)
                        inflight[pool.submit(_work, nxt)] = nxt
        except BaseException:
            # Same actionable resume note the process pool prints, with the ONE invariant
            # that genuinely differs spelled out: the crash window is the whole in-flight
            # window here, not one item per worker. Printed rather than wrapped, so the
            # original exception (and its type) still propagates untouched.
            pair_ids = sorted(task["item"]["pair_id"] for task in inflight.values())
            print(
                (f"{len(pair_ids)} item(s) were in flight and are NOT persisted "
                 f"({pair_ids[:5]}); every other completed record is on disk. Re-run "
                 "`--generate-grounded` to resume: up to --generate-grounded-concurrency "
                 "in-flight items are redone."
                 if pair_ids else
                 "No item was in flight; every completed record is already persisted."),
                file=sys.stderr,
            )
            for future in inflight:
                future.cancel()
            raise


def _grounded_inflight_note(inflight, by_slot):
    """Name the items that were in flight, so a resume message is actionable."""
    if not inflight:
        return "No item was in flight; every completed record is already persisted."
    pair_ids = sorted(by_slot[slot]["item"]["pair_id"] for slot in inflight.values())
    return (
        f"{len(pair_ids)} item(s) were in flight and are NOT persisted "
        f"({pair_ids[:5]}); every other completed record is on disk. Re-run "
        "`--generate-grounded` to resume: at most one unpersisted in-flight item per "
        "crashed worker is redone."
    )


def _load_grounded_split(split, args):
    """Rebuild ONE split's base rows from source (leak checks + drift guard) for the
    grounded paths. Returns (items, base_rows, base_jsonl, split_name)."""
    items, base_rows, _, _, split_name = _load_split_base_rows(split, args)
    base_jsonl, _, _ = _grounded_paths(split, args)
    return items, base_rows, base_jsonl, split_name


def check_grounded(args):
    """--check-grounded (torch-free): verify BOTH split artifacts with the training-time
    loader and print the audit report."""
    for split in ("train", "eval"):
        cfg = mode_config(args)
        items, base_rows, _, split_name = _load_grounded_split(split, args)
        stories_by_title = load_stories_for_items(items, cfg)
        analyses = load_and_verify_grounded_records(
            base_rows, items, stories_by_title, split, args)
        _, grounded_path, _ = _grounded_paths(split, args)
        with open(grounded_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        resolved = [r for r in records if r["status"] == "resolved"]
        spans = sorted(e["span_norm_chars"] for r in resolved
                       for e in r["evidence"].values()) or [0]
        lengths = sorted(r["analysis_chars"] for r in resolved) or [0]
        verified = {d: sum(1 for r in resolved if r["evidence"][d]["verified"])
                    for d in GROUNDED_DEBATERS}
        organic = [(r["pair_id"], r["audit"]["organic_h_overlap"]) for r in resolved
                   if r["audit"]["organic_h_overlap"]]
        print(f"--check-grounded [{split}]: {len(records)} items "
              f"({len(resolved)} resolved), span norm chars min/median/max = "
              f"{spans[0]}/{spans[len(spans) // 2]}/{spans[-1]}, analysis chars "
              f"min/median/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}.")
        print(f"  verified spans: Debater A {verified['Debater A']}/{len(resolved)}, "
              f"Debater B {verified['Debater B']}/{len(resolved)}")
        if organic:
            print(f"  organic H-text overlap in analysis (audited, allowed — also in the "
                  f"transcript): {len(organic)}")
            for pair_id, hits in organic[:10]:
                print(f"    {pair_id}  {hits}")
        for record in records:
            if record["status"] == "resolved":
                continue
            print(f"  UNRESOLVED {record['pair_id']}: "
                  + "; ".join(f"[{f['tier']}#{f['attempt']}] {f['reason']}"
                              for f in record["failures"]))
        # Orientation invariance: one analysis per ITEM feeds both rows by construction.
        _, rows_by_index = group_grounded_rows_by_item(base_rows)
        for dataset_index, analysis in analyses.items():
            if analysis is None:
                continue
            rows = rows_by_index[dataset_index]
            continuations = {
                o: build_grounded_continuation(analysis, rows[o]["target"])
                for o in ORIENTATIONS
            }
            if (continuations["A=Y_true"].replace("Answer: A", "Answer: ?")
                    != continuations["A=Y_false"].replace("Answer: B", "Answer: ?")):
                raise RuntimeError(
                    f"orientation invariance violated for item {dataset_index}: the two "
                    "continuations differ by more than the final answer letter"
                )
        print(f"  orientation invariance verified on {len(analyses)} items.")
    print("--check-grounded complete: both splits verified.")


def resolve_grounded(base_rows, split, items, args):
    """Attach row['grounded_analysis'] (None for fallback rows) and
    row['grounded_lm_enabled'] to every row. Mirrors resolve_rationales; loads stories
    itself so no existing signature changes."""
    cfg = mode_config(args)
    stories_by_title = load_stories_for_items(items, cfg)
    analyses = load_and_verify_grounded_records(
        base_rows, items, stories_by_title, split, args)
    stale = [r["row_id"] for r in base_rows if "rationale" in r]
    if stale:
        raise RuntimeError(
            f"grounded rows unexpectedly carry rationale content (e.g. {stale[:3]}); "
            "refusing to mix supervision objectives."
        )
    fallback = 0
    for row in base_rows:
        analysis = analyses[row["dataset_index"]]
        row["grounded_analysis"] = analysis
        row["grounded_lm_enabled"] = analysis is not None
        if analysis is None:
            fallback += 1
    print(f"[grounded] {split}: {len(base_rows) - fallback}/{len(base_rows)} rows carry an "
          f"audited grounded analysis; {fallback} fallback row(s) train the direct "
          "forced-choice view only.")
    return fallback


def grounded_artifact_provenance_metadata(cfg, args, artifact_train_records,
                                          artifact_eval_records, training_indices,
                                          eval_indices, train_all):
    """Artifact-vs-training-set digests for checkpoint metadata, with the same
    pre-merge-vs-post-selection discipline as qh_artifact_provenance_metadata:
    *_artifact_sha256 always matches the corresponding file; training_set_sha256 always
    matches the records the Trainer actually consumed."""
    by_index = {r["dataset_index"]: r for r in artifact_train_records}
    eval_by_index = {r["dataset_index"]: r for r in artifact_eval_records}
    merged = dict(by_index)
    merged.update(eval_by_index)
    training_records = [merged[i] for i in training_indices if i in merged]
    trainer_eval_records = [merged[i] for i in eval_indices if i in merged]
    return {
        "train_artifact": cfg["grounded_train_jsonl"],
        "eval_artifact": cfg["grounded_eval_jsonl"],
        "train_manifest": cfg["grounded_manifest_template"].format(split="train"),
        "eval_manifest": cfg["grounded_manifest_template"].format(split="eval"),
        "train_artifact_items": len(artifact_train_records),
        "eval_artifact_items": len(artifact_eval_records),
        "train_artifact_sha256": grounded_combined_hash(artifact_train_records),
        "eval_artifact_sha256": grounded_combined_hash(artifact_eval_records),
        "training_set_items": len(training_records),
        "training_set_sha256": grounded_combined_hash(training_records),
        "trainer_eval_set_items": len(trainer_eval_records),
        "train_all_uses_eval_artifact_as_training_data": bool(train_all),
        "combined_hash_policy": (
            "sha256 over the newline-joined per-record grounded_content_sha256 chain "
            "(exact-key-set coverage minus GROUNDED_VOLATILE_FIELDS)"
        ),
    }


# ---------------------------------------------------------------------------
# Tokenization helpers (cluster; tokenizer-only, no torch required)
# ---------------------------------------------------------------------------

def build_supervised_continuation(rationale, target):
    """Continuation generated after the chat-template generation prompt.

    The base gpt-oss judge emits analysis first, then starts a new assistant final
    channel. We supervise the whole transition: the per-example transcript-grounded
    `rationale` in the analysis channel, the final-channel header, "Answer: X", and the
    return token. Production ``rationale`` content is per-example and non-constant;
    ``ANALYSIS_TARGET`` is reserved for explicit smoke/tokenizer validation.
    """
    return (
        f"{ANALYSIS_CHANNEL_PREFIX}{rationale}"
        f"{FINAL_CHANNEL_PREFIX}{target}{RETURN_TOKEN}"
    )


def _answer_only_continuation(target):
    """Return the canonical answer-only continuation bytes."""
    return f"{FINAL_CHANNEL_HEADER}{target}{RETURN_TOKEN}"


def build_answer_only_continuation(target):
    """Direct Harmony final-channel continuation with no analysis channel, channel
    transition, rationale, or other semantic content."""
    if target not in {"Answer: A", "Answer: B"}:
        raise RuntimeError(f"unexpected answer-only target {target!r}; expected Answer: A/B")
    return _answer_only_continuation(target)


def build_grounded_continuation(analysis, target):
    """Grounded continuation: the SAME harmony scaffold the rationale arm uses, carrying
    the audited, orientation-invariant grounded analysis instead of a rationale. The
    scaffold bytes are shared deliberately (one harmony contract, one parser); only the
    analysis payload and its audit differ."""
    if target not in {"Answer: A", "Answer: B"}:
        raise RuntimeError(f"unexpected grounded target {target!r}; expected Answer: A/B")
    if not isinstance(analysis, str) or not analysis:
        raise RuntimeError("grounded supervision requires a non-empty analysis")
    return build_supervised_continuation(analysis, target)


def build_continuation(supervision, rationale, target, grounded_analysis=None):
    """Dispatch to the exact continuation for the selected supervision arm."""
    if supervision == SUPERVISION_RATIONALE:
        if rationale is None:
            raise RuntimeError("rationale supervision requires a non-null rationale")
        if grounded_analysis is not None:
            raise RuntimeError("rationale supervision received a grounded analysis")
        return build_supervised_continuation(rationale, target)
    if supervision == SUPERVISION_ANSWER_ONLY:
        if rationale is not None:
            raise RuntimeError(
                "answer-only supervision received a rationale; refusing to ignore or "
                "teacher-force hidden CoT"
            )
        if grounded_analysis is not None:
            raise RuntimeError("answer-only supervision received a grounded analysis")
        return build_answer_only_continuation(target)
    if supervision == SUPERVISION_GROUNDED:
        if rationale is not None:
            raise RuntimeError(
                "grounded supervision received a rationale; refusing to mix objectives"
            )
        if grounded_analysis is None:
            raise RuntimeError("grounded supervision requires a verified grounded analysis")
        return build_grounded_continuation(grounded_analysis, target)
    raise RuntimeError(f"unknown supervision objective {supervision!r}")


def supervised_continuation_template(supervision):
    """Human-readable, literal continuation contract for checkpoint metadata."""
    target = "Answer: <A|B>"
    if supervision == SUPERVISION_RATIONALE:
        return build_supervised_continuation("<rationale>", target)
    if supervision == SUPERVISION_ANSWER_ONLY:
        return _answer_only_continuation(target)
    if supervision == SUPERVISION_GROUNDED:
        return build_supervised_continuation("<grounded analysis>", target)
    raise RuntimeError(f"unknown supervision objective {supervision!r}")


def harmony_control_ids(tokenizer):
    """Return canonical single token IDs for all Harmony control tokens."""
    ids_by_token = {}
    for tok in HARMONY_CONTROL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        expected = tokenizer.convert_tokens_to_ids(tok)
        if len(ids) != 1 or ids[0] != expected:
            raise RuntimeError(
                f"harmony control token {tok!r} does not map to one canonical ID: "
                f"encode={ids}, convert_tokens_to_ids={expected!r}"
            )
        ids_by_token[tok] = expected
    if len(set(ids_by_token.values())) != len(HARMONY_CONTROL_TOKENS):
        raise RuntimeError(
            "Harmony control tokens do not map to mutually distinct canonical IDs: "
            f"{ids_by_token!r}"
        )
    return ids_by_token


def tokenize_with_target_mask(
    messages,
    rationale,
    tokenizer,
    max_len,
    supervision=SUPERVISION_RATIONALE,
    control_ids=None,
    grounded_analysis=None,
):
    """Render the user prompt with the normal generation prompt, then append and
    supervise a verifier continuation while masking the full user prompt.

    In rationale mode, loss flows through the transcript-grounded rationale plus
    the final-channel transition and verdict. In answer-only mode, loss flows only
    through the direct final-channel format scaffold, "Answer: A/B", and return
    token. In grounded mode, loss flows through the audited grounded analysis and the
    same final-channel transition and verdict. The prompt is always fully masked.
    """
    if control_ids is None:
        control_ids = harmony_control_ids(tokenizer)
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    if prefix_text.endswith(ANALYSIS_CHANNEL_PREFIX):
        raise RuntimeError(
            "Chat template already prefilled the analysis channel opener; do not "
            "also prepend ANALYSIS_CHANNEL_PREFIX in the supervised continuation."
        )
    if not prefix_text.endswith(GENERATION_PROMPT_SUFFIX):
        raise RuntimeError(
            "Unexpected chat-template generation prompt suffix; expected it to end "
            f"with {GENERATION_PROMPT_SUFFIX!r}, got tail {prefix_text[-120:]!r}. "
            "Revisit the supervised harmony continuation before training."
        )
    target = messages[-1]["content"]
    if target not in {"Answer: A", "Answer: B"}:
        raise RuntimeError(f"unexpected assistant target {target!r}; expected Answer: A/B")
    continuation = build_continuation(supervision, rationale, target, grounded_analysis)
    full_text = prefix_text + continuation

    if not full_text.startswith(prefix_text):
        raise RuntimeError("internal error: constructed full_text is not prefixed by prefix_text")

    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids

    prefix_len = 0
    while (
        prefix_len < len(prefix_ids)
        and prefix_len < len(full_ids)
        and prefix_ids[prefix_len] == full_ids[prefix_len]
    ):
        prefix_len += 1
    if prefix_len != len(prefix_ids):
        raise RuntimeError(
            "Tokenized prefix is not a token prefix of the constructed full text; "
            "revisit supervised continuation construction."
        )

    # Token-level continuation integrity for EVERY row: the
    # supervised span must equal the standalone re-tokenization of the continuation.
    # Both arms begin at the single-ID <|channel|> control token, so there is no
    # boundary merge with the prompt.
    if full_ids[prefix_len:] != cont_ids:
        raise RuntimeError(
            "Supervised continuation token span does not match the standalone "
            f"tokenization of the intended continuation (row target {target!r}); a "
            "tokenizer boundary effect or a control token inside the rationale would "
            "cause this."
        )

    channel_id = control_ids["<|channel|>"]
    return_id = control_ids["<|return|>"]
    end_id = control_ids["<|end|>"]
    start_id = control_ids["<|start|>"]
    if not cont_ids or cont_ids[0] != channel_id:
        raise RuntimeError(
            f"{supervision} continuation must start with canonical <|channel|> token"
        )
    if cont_ids[-1] != return_id:
        raise RuntimeError(
            f"{supervision} continuation must end with canonical <|return|> token"
        )
    if supervision == SUPERVISION_ANSWER_ONLY:
        if (
            cont_ids.count(channel_id) != 1
            or end_id in cont_ids
            or start_id in cont_ids
            or len(cont_ids) > ANSWER_ONLY_MAX_CONT_TOKENS
        ):
            raise RuntimeError(
                "answer-only continuation contains an analysis/channel-transition "
                "structure or is unexpectedly long; expected exactly one direct final "
                f"channel and <= {ANSWER_ONLY_MAX_CONT_TOKENS} tokens"
            )
    elif supervision in (SUPERVISION_RATIONALE, SUPERVISION_GROUNDED):
        if (
            cont_ids.count(channel_id) != 2
            or end_id not in cont_ids
            or start_id not in cont_ids
        ):
            raise RuntimeError(
                f"{supervision} continuation lacks the expected analysis-to-final "
                "Harmony channel transition"
            )
        if (supervision == SUPERVISION_GROUNDED
                and len(cont_ids) > GROUNDED_MAX_CONT_TOKENS):
            raise RuntimeError(
                f"grounded continuation is {len(cont_ids)} tokens > "
                f"{GROUNDED_MAX_CONT_TOKENS}; the audited analysis length cap "
                f"({GROUNDED_MAX_CHARS} chars) should make this impossible"
            )
    else:
        raise RuntimeError(f"unknown supervision objective {supervision!r}")

    if len(full_ids) > max_len:
        return None

    labels = list(full_ids)
    for i in range(prefix_len):
        labels[i] = -100
    if all(l == -100 for l in labels):
        return None
    kept = [t for t, label in zip(full_ids, labels) if label != -100]
    if kept != cont_ids:
        raise RuntimeError(
            "unmasked label token IDs do not exactly equal the intended continuation "
            f"for {supervision} supervision"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "continuation_ids": cont_ids,
    }


class PadCollator:
    """Right-pad a batch of pre-tokenized examples. Pad tokens get attention_mask 0
    and label -100 so they never contribute to the loss."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _check_unmasked_label_spans(
    ds,
    tokenizer,
    n=3,
    seed=0,
    supervision=SUPERVISION_RATIONALE,
):
    """Audit that the unmasked label span exactly equals the intended continuation.

    The span must match the
    intended continuation token IDs (carried on the record as `continuation_ids`), plus
    structural sanity. Decoded text additionally proves that answer-only has no analysis
    channel and no semantic content besides the A/B verdict."""

    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    channel_id = harmony_control_ids(tokenizer)["<|channel|>"]
    for i in idxs:
        sample = ds[i]
        kept = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
        if not kept:
            raise RuntimeError(f"example {i}: all labels are -100 — mask swallowed the target.")
        if len(kept) == len(sample["input_ids"]):
            raise RuntimeError(f"example {i}: nothing was masked — the prompt prefix wasn't detected.")
        cont_ids = sample.get("continuation_ids")
        if cont_ids is None:
            raise RuntimeError(f"example {i}: record is missing continuation_ids for the mask audit.")
        if kept != list(cont_ids):
            raise RuntimeError(
                f"example {i}: unmasked label token IDs ({len(kept)}) do not equal the "
                f"intended continuation IDs ({len(cont_ids)}); the mask is misaligned with "
                "the supervised continuation."
            )
        if kept[0] != channel_id:
            raise RuntimeError(
                f"example {i}: first supervised token id {kept[0]} != <|channel|> id "
                f"{channel_id}; {supervision} channel opener is malformed."
            )
        kept_text = tokenizer.decode(kept, skip_special_tokens=False)
        if FINAL_MARKER not in kept_text:
            raise RuntimeError(
                f"example {i}: unmasked span lacks the final-channel marker {FINAL_MARKER!r}. "
                f"Span: {kept_text[:200]!r}"
            )
        if not re.search(r"Answer: [AB]\b", kept_text):
            raise RuntimeError(
                f"example {i}: unmasked span lacks an 'Answer: A/B' verdict. Span: {kept_text[:200]!r}"
            )
        if supervision == SUPERVISION_ANSWER_ONLY:
            if ANALYSIS_CHANNEL_PREFIX in kept_text:
                raise RuntimeError(
                    f"example {i}: answer-only span contains an analysis-channel opener"
                )
            if not kept_text.startswith(FINAL_CHANNEL_HEADER) or not kept_text.endswith(RETURN_TOKEN):
                raise RuntimeError(
                    f"example {i}: answer-only span is not a direct final-channel "
                    f"continuation: {kept_text!r}"
                )
            semantic = kept_text[len(FINAL_CHANNEL_HEADER):-len(RETURN_TOKEN)]
            if re.fullmatch(r"Answer: [AB]", semantic) is None:
                raise RuntimeError(
                    f"example {i}: answer-only semantic payload is not exactly "
                    f"'Answer: A/B': {semantic!r}"
                )
            print(
                f"Mask check OK (example {i}): {len(kept)}/{len(sample['input_ids'])} "
                f"tokens trained; span={kept_text!r}"
            )
        elif supervision in (SUPERVISION_RATIONALE, SUPERVISION_GROUNDED):
            if not kept_text.startswith(ANALYSIS_CHANNEL_PREFIX):
                # Grounded fallback rows (unresolved items under --grounded-unresolved
                # answer-only) legitimately carry the answer-only continuation instead.
                if supervision == SUPERVISION_GROUNDED and kept_text.startswith(
                    FINAL_CHANNEL_HEADER
                ):
                    print(
                        f"Mask check OK (example {i}): {len(kept)}/"
                        f"{len(sample['input_ids'])} tokens trained; grounded FALLBACK row "
                        f"(direct final channel only): {kept_text!r}"
                    )
                    continue
                raise RuntimeError(
                    f"example {i}: {supervision} span does not open the analysis channel"
                )
            analysis = kept_text.split(ANALYSIS_CHANNEL_PREFIX, 1)[-1].split("<|end|>", 1)[0]
            if supervision == SUPERVISION_GROUNDED:
                if "Answer:" in analysis:
                    raise RuntimeError(
                        f"example {i}: grounded analysis contains 'Answer:'; the verdict "
                        "belongs only in the final channel and the analysis must be "
                        "orientation-invariant"
                    )
                assert_rendered_analysis_shape(analysis)
            print(
                f"Mask check OK (example {i}): {len(kept)}/{len(sample['input_ids'])} "
                f"tokens trained; analysis[:70]={analysis.strip()[:70]!r}"
            )
        else:
            raise RuntimeError(f"unknown supervision objective {supervision!r}")


def tokenize_rows(
    rows,
    tokenizer,
    max_len,
    supervision=SUPERVISION_RATIONALE,
):
    """Tokenize with the supervised harmony-continuation mask. ANY drop is data
    corruption (prompts are ~2k tokens << max_len) and silently unbalances paired
    orientations — so we fail, not warn."""
    tokenized, dropped = [], []
    control_ids = harmony_control_ids(tokenizer)
    for row in rows:
        if supervision == SUPERVISION_RATIONALE and "rationale" not in row:
            raise RuntimeError(f"row {row['row_id']} has no resolved rationale — call "
                               "resolve_rationales() before tokenize_rows().")
        if supervision != SUPERVISION_RATIONALE and "rationale" in row:
            raise RuntimeError(
                f"{supervision} row {row['row_id']} unexpectedly carries a rationale; "
                "refusing to risk hidden CoT supervision"
            )
        if supervision != SUPERVISION_GROUNDED and "grounded_analysis" in row:
            raise RuntimeError(
                f"{supervision} row {row['row_id']} unexpectedly carries a grounded "
                "analysis; refusing to mix supervision objectives"
            )
        if supervision == SUPERVISION_GROUNDED and "grounded_analysis" not in row:
            raise RuntimeError(f"row {row['row_id']} has no resolved grounded analysis — "
                               "call resolve_grounded() before tokenize_rows().")
        rationale = row.get("rationale") if supervision == SUPERVISION_RATIONALE else None
        grounded_analysis = (row.get("grounded_analysis")
                             if supervision == SUPERVISION_GROUNDED else None)
        # A grounded item the teacher never resolved falls back to the answer-only
        # continuation and contributes ONLY the direct forced-choice view (its LM forward
        # is skipped in GroundedTrainer); the row is still a first-class, well-formed row.
        row_supervision = supervision
        if supervision == SUPERVISION_GROUNDED and grounded_analysis is None:
            row_supervision = SUPERVISION_ANSWER_ONLY
        rec = tokenize_with_target_mask(
            row["messages"],
            rationale,
            tokenizer,
            max_len,
            supervision=row_supervision,
            control_ids=control_ids,
            grounded_analysis=grounded_analysis,
        )
        if rec is None:
            dropped.append(row["row_id"])
        else:
            rec["row_id"] = row["row_id"]
            if supervision == SUPERVISION_GROUNDED:
                rec["grounded_lm_enabled"] = grounded_analysis is not None
            tokenized.append(rec)
    if dropped:
        raise RuntimeError(
            f"{len(dropped)} rows exceeded max_len={max_len} (e.g. {dropped[:3]}) — "
            "this should be impossible for these verifier judge prompts and would break "
            "the paired-orientation balance. Investigate before training."
        )
    return tokenized


# ---------------------------------------------------------------------------
# Q_H auxiliary tokenization (cluster; tokenizer-only, no torch required)
# ---------------------------------------------------------------------------

def qh_letter_token_ids(tokenizer):
    """Canonical single token ids for the scored completions " A" / " B"."""
    ids_by_letter = {}
    for letter, completion in LABEL_COMPLETIONS.items():
        ids = list(tokenizer(completion, add_special_tokens=False).input_ids)
        if len(ids) != 1:
            raise RuntimeError(
                f"Q_H readout completion {completion!r} does not map to exactly one token "
                f"(ids={ids}); the restricted two-letter readout would be ill-defined."
            )
        ids_by_letter[letter] = int(ids[0])
    if len(set(ids_by_letter.values())) != len(ids_by_letter):
        raise RuntimeError(f"' A' and ' B' map to the same token id: {ids_by_letter!r}")
    return ids_by_letter


def qh_prefill_token_ids(tokenizer, control_ids=None):
    """Token ids of the harmony final-channel prefill, structurally validated."""
    if control_ids is None:
        control_ids = harmony_control_ids(tokenizer)
    ids = list(tokenizer(FINAL_CHANNEL_PREFILL, add_special_tokens=False).input_ids)
    if not ids:
        raise RuntimeError("empty tokenization for the final-channel prefill scaffold")
    if ids[0] != control_ids["<|channel|>"]:
        raise RuntimeError(
            f"final-channel prefill does not start with the canonical <|channel|> id "
            f"(ids={ids}); the readout would be scored at a mis-placed slot."
        )
    if control_ids["<|message|>"] not in ids:
        raise RuntimeError(
            f"final-channel prefill lost its canonical <|message|> id (ids={ids})."
        )
    return [int(i) for i in ids]


def assert_qy_answer_slot_letter_tokens(tokenizer, letter_ids, control_ids=None,
                                        prefill_ids=None):
    """The aux loss MUST score the same token the answer-only arm supervises.

    Tokenizes the real answer-only continuation for both targets and checks that the
    token immediately before <|return|> is exactly the ' A'/' B' id used by the Q_H
    readout. Returns the (constant) answer-only continuation length C.
    """
    if control_ids is None:
        control_ids = harmony_control_ids(tokenizer)
    lengths = set()
    for letter, token_id in letter_ids.items():
        continuation = build_answer_only_continuation(TARGET_TEMPLATE.format(letter=letter))
        cont_ids = list(tokenizer(continuation, add_special_tokens=False).input_ids)
        if len(cont_ids) < 2 or cont_ids[-1] != control_ids["<|return|>"]:
            raise RuntimeError(
                f"answer-only continuation for {letter!r} does not end with <|return|> "
                f"(ids={cont_ids})"
            )
        if int(cont_ids[-2]) != int(token_id):
            raise RuntimeError(
                f"Q_Y answer-slot letter token for {letter!r} is {cont_ids[-2]}, but the Q_H "
                f"readout would score {token_id}. The auxiliary loss must act on the SAME "
                "output symbol the Q_Y arm supervises."
            )
        if prefill_ids is not None:
            expected = list(prefill_ids) + [int(token_id), control_ids["<|return|>"]]
            if cont_ids != expected:
                raise RuntimeError(
                    f"Q_Y answer-only continuation for {letter!r} is not exactly the "
                    "forced-choice final-channel prefill followed by its single letter and "
                    f"<|return|>: got {cont_ids}, expected {expected}."
                )
        lengths.add(len(cont_ids))
    if len(lengths) != 1:
        raise RuntimeError(
            f"answer-only continuation length differs between A and B ({sorted(lengths)}); "
            "the pure-arm loss normalization would depend on the label."
        )
    return lengths.pop()


def count_subsequence(haystack, needle):
    """Number of (overlapping) occurrences of `needle` inside `haystack`."""
    needle = list(needle)
    if not needle:
        raise ValueError("needle must be non-empty")
    haystack = list(haystack)
    n = len(needle)
    return sum(1 for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle)


def assemble_readout_input_ids(prompt_ids, prefill_ids, max_len, letter_token_ids=(),
                               label="forced-choice"):
    """Pure ID-level assembly shared by Q_Y and Q_H forced-choice readouts.

    input = chat-template user generation prompt IDs ++ final-channel prefill IDs.
    Fails CLOSED on length (a dropped leg would silently unbalance the pairs),
    asserts the prefill is the suffix and occurs EXACTLY ONCE, and asserts that no
    answer letter was appended.
    """
    prompt_ids = [int(i) for i in prompt_ids]
    prefill_ids = [int(i) for i in prefill_ids]
    if not prefill_ids:
        raise RuntimeError("empty final-channel prefill ids")
    if not prompt_ids:
        raise RuntimeError(f"empty {label} prompt ids")
    ids = prompt_ids + prefill_ids
    if len(ids) > max_len:
        raise RuntimeError(
            f"{label} readout input is {len(ids)} tokens > max_len={max_len}. Dropping it would "
            "unbalance the paired Q_Y/Q_H legs, so this fails closed. Investigate before "
            "training."
        )
    if ids[-len(prefill_ids):] != prefill_ids:
        raise RuntimeError(
            f"final-channel prefill is not the tokenized suffix of the {label} input"
        )
    occurrences = count_subsequence(ids, prefill_ids)
    if occurrences != 1:
        raise RuntimeError(
            f"final-channel prefill appears {occurrences} times in the {label} input; it must "
            "appear exactly once, at the very end (a duplicate means the scaffold leaked "
            "into the user prompt)."
        )
    for token_id in letter_token_ids:
        if ids[-1] == int(token_id):
            raise RuntimeError(
                f"an answer letter token is appended to the {label} input; the target letter is "
                "read from the logits, never teacher-forced."
            )
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def assemble_qh_input_ids(prompt_ids, prefill_ids, max_len, letter_token_ids=()):
    """Return the shared readout assembly under Q_H-specific field names."""
    rec = assemble_readout_input_ids(
        prompt_ids, prefill_ids, max_len, letter_token_ids, label="Q_H"
    )
    return {
        "qh_input_ids": rec["input_ids"],
        "qh_attention_mask": rec["attention_mask"],
    }


def _as_id_list(encoded):
    """Normalize apply_chat_template(tokenize=True) output to a flat list of ints."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise RuntimeError(f"expected a single tokenized conversation, got {len(encoded)}")
        encoded = encoded[0]
    return [int(i) for i in encoded]


def attach_qy_forced_choice_readouts(rows, tokenized_rows, prefill_ids, letter_ids,
                                     return_token_id, max_len):
    """Attach a no-letter Q_Y readout derived from each audited tokenized training row.

    The answer-only continuation is structurally pinned to
    FINAL_CHANNEL_PREFILL ++ [target letter] ++ [RETURN_TOKEN]. Removing exactly the
    final two tokens therefore exposes the same answer slot used by Q_H without
    re-rendering the prompt or independently deriving its target.
    """
    if len(rows) != len(tokenized_rows):
        raise RuntimeError(
            f"Q_Y forced-choice attachment got {len(rows)} source rows but "
            f"{len(tokenized_rows)} tokenized rows"
        )
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("duplicate Q_Y row ids while attaching forced-choice readouts")
    inverse_letter_ids = {int(token_id): letter for letter, token_id in letter_ids.items()}
    if len(inverse_letter_ids) != 2:
        raise RuntimeError(f"Q_Y forced-choice letter ids are not distinct: {letter_ids!r}")

    for rec in tokenized_rows:
        row_id = rec.get("row_id")
        row = by_id.get(row_id)
        if row is None:
            raise RuntimeError(f"tokenized Q_Y row {row_id!r} has no audited source row")
        full_ids = [int(i) for i in rec["input_ids"]]
        cont_ids = [int(i) for i in rec.get("continuation_ids", ())]
        if len(cont_ids) != len(prefill_ids) + QY_READOUT_TAIL_TOKENS:
            raise RuntimeError(
                f"Q_Y row {row_id} continuation has {len(cont_ids)} tokens; expected "
                f"len(prefill) + {QY_READOUT_TAIL_TOKENS} = "
                f"{len(prefill_ids) + QY_READOUT_TAIL_TOKENS}"
            )
        if cont_ids[:-QY_READOUT_TAIL_TOKENS] != list(prefill_ids):
            raise RuntimeError(
                f"Q_Y row {row_id} continuation does not begin with the exact "
                "FINAL_CHANNEL_PREFILL token ids"
            )
        if cont_ids[-1] != int(return_token_id):
            raise RuntimeError(
                f"Q_Y row {row_id} continuation does not end with the canonical return token"
            )
        target_token_id = cont_ids[-2]
        target_letter = inverse_letter_ids.get(target_token_id)
        if target_letter is None:
            raise RuntimeError(
                f"Q_Y row {row_id} answer-slot token {target_token_id} is neither ' A' nor ' B'"
            )
        if row.get("target_letter") != target_letter:
            raise RuntimeError(
                f"Q_Y row {row_id} token-derived target {target_letter!r} disagrees with "
                f"audited row target {row.get('target_letter')!r}"
            )
        if len(full_ids) < len(cont_ids) or full_ids[-len(cont_ids):] != cont_ids:
            raise RuntimeError(
                f"Q_Y row {row_id} input_ids do not end with their audited continuation ids"
            )
        prompt_ids = full_ids[:-len(cont_ids)]
        readout = assemble_readout_input_ids(
            prompt_ids,
            prefill_ids,
            max_len,
            letter_token_ids=tuple(letter_ids.values()),
            label="Q_Y",
        )
        expected_readout_ids = full_ids[:-QY_READOUT_TAIL_TOKENS]
        if readout["input_ids"] != expected_readout_ids:
            raise RuntimeError(
                f"Q_Y row {row_id} derived readout is not exactly input_ids without its "
                "target letter and return token"
            )
        rec["qy_readout_input_ids"] = readout["input_ids"]
        rec["qy_readout_attention_mask"] = readout["attention_mask"]
        rec["qy_target_index"] = 0 if target_letter == "A" else 1
        rec["qy_target_letter"] = target_letter
    return tokenized_rows


def attach_grounded_direct_readouts(rows, tokenized_rows, prefill_ids, letter_ids,
                                    return_token_id, max_len):
    """Attach the DIRECT (analysis-free) Q_Y forced-choice readout to grounded rows.

    Unlike the answer-only arm — where the readout is the audited input minus its last two
    tokens — a grounded row's continuation carries the analysis channel, so the readout is
    rebuilt as `chat prompt ids ++ FINAL_CHANNEL_PREFILL ids`: the model must commit to a
    letter with NO analysis in context. The prompt prefix is taken from the audited
    tokenized row (tokenize_with_target_mask already proved it is a token prefix of the
    full input), and the target letter comes from the audited row, cross-checked against
    the token the LM view supervises at the answer slot.
    """
    if len(rows) != len(tokenized_rows):
        raise RuntimeError(
            f"grounded direct-readout attachment got {len(rows)} source rows but "
            f"{len(tokenized_rows)} tokenized rows"
        )
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("duplicate row ids while attaching grounded direct readouts")
    inverse_letter_ids = {int(token_id): letter for letter, token_id in letter_ids.items()}
    if len(inverse_letter_ids) != 2:
        raise RuntimeError(f"grounded letter ids are not distinct: {letter_ids!r}")
    prefill_ids = [int(i) for i in prefill_ids]

    for rec in tokenized_rows:
        row_id = rec.get("row_id")
        row = by_id.get(row_id)
        if row is None:
            raise RuntimeError(f"tokenized grounded row {row_id!r} has no audited source row")
        full_ids = [int(i) for i in rec["input_ids"]]
        cont_ids = [int(i) for i in rec.get("continuation_ids", ())]
        if not cont_ids or len(full_ids) <= len(cont_ids):
            raise RuntimeError(f"grounded row {row_id} has no usable continuation ids")
        if full_ids[-len(cont_ids):] != cont_ids:
            raise RuntimeError(
                f"grounded row {row_id} input_ids do not end with their audited "
                "continuation ids"
            )
        target_letter = row.get("target_letter")
        if target_letter not in letter_ids:
            raise RuntimeError(
                f"grounded row {row_id} has target letter {target_letter!r}"
            )
        # Contract: whatever the analysis, the continuation TAIL is exactly the
        # final-channel prefill, the target letter, and the return token — the slot the
        # production readout scores.
        tail = cont_ids[-(len(prefill_ids) + QY_READOUT_TAIL_TOKENS):]
        expected_tail = prefill_ids + [int(letter_ids[target_letter]), int(return_token_id)]
        if tail != expected_tail:
            raise RuntimeError(
                f"grounded row {row_id} continuation does not end with the exact "
                f"final-channel prefill + letter + return ids: got {tail}, expected "
                f"{expected_tail}"
            )
        token_letter = inverse_letter_ids.get(cont_ids[-2])
        if token_letter != target_letter:
            raise RuntimeError(
                f"grounded row {row_id} token-derived target {token_letter!r} disagrees "
                f"with audited row target {target_letter!r}"
            )
        prompt_ids = full_ids[:-len(cont_ids)]
        readout = assemble_readout_input_ids(
            prompt_ids,
            prefill_ids,
            max_len,
            letter_token_ids=tuple(letter_ids.values()),
            label="grounded Q_Y",
        )
        rec["qy_readout_input_ids"] = readout["input_ids"]
        rec["qy_readout_attention_mask"] = readout["attention_mask"]
        rec["qy_target_index"] = 0 if target_letter == "A" else 1
        rec["qy_target_letter"] = target_letter
    return tokenized_rows


def assert_qy_chat_template_tokenization_parity(rows, tokenized_rows, tokenizer, n=None,
                                                seed=0):
    """The Q_Y string-tokenization prefix must equal chat-template tokenize=True.

    Production forced-choice scoring uses apply_chat_template(tokenize=True). This guard
    proves the already-audited Q_Y training prefix is byte/token equivalent before its
    forced-choice prefill is attached.
    """
    if len(rows) != len(tokenized_rows):
        raise RuntimeError("Q_Y chat-template parity check received misaligned row counts")
    pairs = list(zip(rows, tokenized_rows))
    if n is not None and len(pairs) > n:
        pairs = random.Random(seed).sample(pairs, n)
    for row, rec in pairs:
        if row["row_id"] != rec.get("row_id"):
            raise RuntimeError(
                f"Q_Y chat-template parity row mismatch: {row['row_id']!r} != "
                f"{rec.get('row_id')!r}"
            )
        continuation_ids = list(rec["continuation_ids"])
        prefix_ids = list(rec["input_ids"][:-len(continuation_ids)])
        direct_ids = _as_id_list(tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        ))
        if prefix_ids != direct_ids:
            raise RuntimeError(
                f"Q_Y row {row['row_id']} string-tokenized prompt differs from "
                "apply_chat_template(tokenize=True); refusing a readout-contract mismatch"
            )


def tokenize_qh_row(row, tokenizer, max_len, prefill_ids, letter_ids):
    """Tokenize one Q_H readout row exactly like the production forced-choice scorer."""
    messages = row["messages"]
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise RuntimeError(f"Q_H row {row['row_id']} is not a single user message")
    prefix_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not prefix_text.endswith(GENERATION_PROMPT_SUFFIX):
        raise RuntimeError(
            "Unexpected chat-template generation prompt suffix for the Q_H readout; expected "
            f"it to end with {GENERATION_PROMPT_SUFFIX!r}, got tail {prefix_text[-120:]!r}."
        )
    if prefix_text.endswith(FINAL_CHANNEL_PREFILL):
        raise RuntimeError(
            "Chat template already prefilled the final channel; do not append "
            "FINAL_CHANNEL_PREFILL as well."
        )
    # tokenize=True mirrors ForcedChoiceVerifier.prompt_ids exactly (the production
    # readout never re-tokenizes the rendered string).
    prompt_ids = _as_id_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    rec = assemble_qh_input_ids(prompt_ids, prefill_ids, max_len,
                                letter_token_ids=tuple(letter_ids.values()))
    if row["target_letter"] not in ("A", "B"):
        raise RuntimeError(f"Q_H row {row['row_id']} has target letter {row['target_letter']!r}")
    rec["qh_target_index"] = 0 if row["target_letter"] == "A" else 1
    rec["qh_target_letter"] = row["target_letter"]
    rec["row_id"] = row["row_id"]
    rec["qy_row_id"] = row["qy_row_id"]
    return rec


def tokenize_qh_rows(rows, tokenizer, max_len, prefill_ids, letter_ids):
    return [tokenize_qh_row(row, tokenizer, max_len, prefill_ids, letter_ids) for row in rows]


def _check_readout_spans(records, tokenizer, input_ids_key, target_letter_key,
                         label, n=3, seed=0):
    """Decode sampled forced-choice tails; every input must end before the letter."""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(records)), min(n, len(records)))
    for i in idxs:
        rec = records[i]
        tail_ids = rec[input_ids_key][-24:]
        tail = tokenizer.decode(tail_ids, skip_special_tokens=False)
        if not tail.endswith(FINAL_CHANNEL_PREFILL):
            raise RuntimeError(
                f"{label} example {i} ({rec['row_id']}) does not end at the final-channel answer "
                f"slot; tail={tail!r}"
            )
        if tail.endswith(tuple(LABEL_COMPLETIONS.values())):
            raise RuntimeError(
                f"{label} example {i}: an answer letter is appended; tail={tail!r}"
            )
        print(f"{label} readout check OK (example {i}, {rec['row_id']}): "
              f"{len(rec[input_ids_key])} tokens, target "
              f"{LABEL_COMPLETIONS[rec[target_letter_key]]!r}, tail={tail[-60:]!r}")


def _check_qh_readout_spans(records, tokenizer, n=3, seed=0):
    return _check_readout_spans(
        records, tokenizer, "qh_input_ids", "qh_target_letter", "Q_H", n=n, seed=seed
    )


def _check_qy_readout_spans(records, tokenizer, n=3, seed=0):
    return _check_readout_spans(
        records, tokenizer, "qy_readout_input_ids", "qy_target_letter", "Q_Y",
        n=n, seed=seed,
    )


def constant_continuation_tokens(tokenized_rows):
    """C = the audited answer-only continuation length, which MUST be constant.

    In forced-choice mode C is a structural check proving every Q_Y row has the same
    prefill, letter, and return tail. In token-CE mode it also fixes the loss
    normalization denominator.
    """
    lengths = {len(r["continuation_ids"]) for r in tokenized_rows}
    if len(lengths) != 1:
        raise RuntimeError(
            f"supervised continuation length is not constant ({sorted(lengths)[:5]}); the Q_H "
            "auxiliary arm requires one exact answer-only tail so its Q_Y answer slot can be "
            "derived without re-rendering."
        )
    return lengths.pop()


def check_tokenizer(args):
    from transformers import AutoTokenizer

    # A stale/empty prepared-base directory must not shadow the hub tokenizer during
    # this tokenizer-only audit (train/preflight separately requires a real config.json).
    if os.path.isfile(os.path.join(BF16_DIR, "tokenizer_config.json")):
        source = BF16_DIR
    else:
        source = MODEL_NAME
        if os.path.isdir(BF16_DIR):
            print(
                f"WARNING: {BF16_DIR} exists but has no tokenizer_config.json; "
                f"--check-tokenizer is auditing against {MODEL_NAME} instead. Training "
                "preflight will still reject an incomplete prepared base.",
                file=sys.stderr,
            )
    print(f"Loading tokenizer from {source} ...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    control_ids = harmony_control_ids(tokenizer)

    tokenized = []
    all_base_rows = []
    supervision = supervision_of(args)
    for split in ("train", "eval"):
        items, base_rows, *_ = _load_split_base_rows(split, args)
        if supervision == SUPERVISION_GROUNDED:
            resolve_grounded(base_rows, split, items, args)
        else:
            resolve_rationales(base_rows, split, items, args)
        all_base_rows.extend(base_rows)
        tokenized.extend(
            tokenize_rows(
                base_rows,
                tokenizer,
                args.max_len,
                supervision=supervision,
            )
        )
    lengths = sorted(len(t["input_ids"]) for t in tokenized)
    print(f"Tokenized {len(tokenized)} rows; tokens min/median/max = "
          f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]} (max_len {args.max_len}).")
    _check_unmasked_label_spans(
        tokenized,
        tokenizer,
        n=6,
        supervision=supervision,
    )
    if supervision == SUPERVISION_ANSWER_ONLY:
        mode = "answer-only (no analysis channel, no rationale)"
    elif supervision == SUPERVISION_GROUNDED:
        mode = "per-item audited grounded analysis"
    elif getattr(args, "debug_constant_rationale", False):
        mode = "constant ANALYSIS_TARGET (debug)"
    else:
        mode = "per-example rationale"
    print(f"--check-tokenizer complete: zero drops; supervised spans = {mode} harmony "
          "continuations (token-level mask audit passed).")

    if supervision == SUPERVISION_GROUNDED:
        # ---- grounded direct-view contracts (tokenizer-only, no GPU) ----
        letter_ids = qh_letter_token_ids(tokenizer)          # shared readout helpers:
        prefill_ids = qh_prefill_token_ids(tokenizer, control_ids)  # they only depend on
        assert_qy_answer_slot_letter_tokens(                 # score_verifier constants,
            tokenizer, letter_ids, control_ids, prefill_ids=prefill_ids)  # not on --qh-aux
        attach_grounded_direct_readouts(
            all_base_rows, tokenized, prefill_ids, letter_ids,
            control_ids["<|return|>"], args.max_len,
        )
        assert_qy_chat_template_tokenization_parity(
            all_base_rows, tokenized, tokenizer, n=None
        )
        _check_qy_readout_spans(tokenized, tokenizer, n=4)
        target_dist = Counter(r["qy_target_letter"] for r in tokenized)
        if target_dist["A"] != target_dist["B"]:
            raise RuntimeError(
                f"grounded direct-view targets are not balanced: {dict(target_dist)}"
            )
        cont_lengths = sorted(len(r["continuation_ids"]) for r in tokenized)
        fallback = sum(1 for r in tokenized if not r.get("grounded_lm_enabled", True))
        print(f"--check-tokenizer [grounded] complete: direct-view readouts end at "
              f"{FINAL_CHANNEL_PREFILL!r} with NO analysis in context; ' A'="
              f"{letter_ids['A']} ' B'={letter_ids['B']}; targets A={target_dist['A']} "
              f"B={target_dist['B']}; continuation tokens min/median/max = "
              f"{cont_lengths[0]}/{cont_lengths[len(cont_lengths) // 2]}/"
              f"{cont_lengths[-1]} (cap {GROUNDED_MAX_CONT_TOKENS}); "
              f"{fallback} fallback row(s).")
        if not qh_aux_of(args):
            return

    if not qh_aux_of(args):
        return

    # ---- Q_H auxiliary leg: heavy tokenizer contracts that only run here ----
    cfg = mode_config(args)
    qy_loss = qy_loss_of(args)
    letter_ids = qh_letter_token_ids(tokenizer)
    prefill_ids = qh_prefill_token_ids(tokenizer, control_ids)
    cont_tokens = assert_qy_answer_slot_letter_tokens(
        tokenizer, letter_ids, control_ids, prefill_ids=prefill_ids
    )
    if supervision == SUPERVISION_GROUNDED:
        constant_c = None
        print(f"Grounded+Q_H token contracts OK: ' A'={letter_ids['A']} ' B'={letter_ids['B']} "
              "(single, distinct, identical to the Q_Y answer-slot letter tokens); "
              f"Q_H prefill {FINAL_CHANNEL_PREFILL!r} -> {prefill_ids}; the grounded LM "
              "continuation is intentionally not treated as the answer-only C-token span.")
    else:
        constant_c = constant_continuation_tokens(tokenized)
        if constant_c != cont_tokens:
            raise RuntimeError(
                f"answer-only continuation is {constant_c} tokens on real rows but "
                f"{cont_tokens} on the reference render; the loss normalization is not pinned."
            )
        print(f"Q_Y/Q_H token contracts OK: ' A'={letter_ids['A']} ' B'={letter_ids['B']} (single, "
              f"distinct, identical to the Q_Y answer-slot letter tokens); prefill "
              f"{FINAL_CHANNEL_PREFILL!r} -> {prefill_ids}; answer-only continuation is a constant "
              f"C={constant_c} tokens.")

    if qy_loss == QY_LOSS_FORCED_CHOICE:
        if supervision != SUPERVISION_GROUNDED:
            attach_qy_forced_choice_readouts(
                all_base_rows, tokenized, prefill_ids, letter_ids,
                control_ids["<|return|>"], args.max_len,
            )
        # This is cheap relative to loading the tokenizer and proves every audited row
        # has the same token prefix as production apply_chat_template(tokenize=True).
        assert_qy_chat_template_tokenization_parity(
            all_base_rows, tokenized, tokenizer, n=None
        )
        _check_qy_readout_spans(tokenized, tokenizer, n=4)
        qy_target_dist = Counter(r["qy_target_letter"] for r in tokenized)
        if qy_target_dist["A"] != qy_target_dist["B"]:
            raise RuntimeError(
                f"Q_Y forced-choice targets are not balanced: {dict(qy_target_dist)}"
            )

    qh_train, qh_eval, _ = build_qh_all(args, write_artifacts=False)
    verify_qh_artifacts_match(qh_train, cfg["qh_train_jsonl"])
    verify_qh_artifacts_match(qh_eval, cfg["qh_eval_jsonl"])
    qh_tokenized = tokenize_qh_rows(qh_train + qh_eval, tokenizer, args.max_len,
                                    prefill_ids, letter_ids)
    qh_lengths = sorted(len(r["qh_input_ids"]) for r in qh_tokenized)
    print(f"Tokenized {len(qh_tokenized)} Q_H rows; tokens min/median/max = "
          f"{qh_lengths[0]}/{qh_lengths[len(qh_lengths) // 2]}/{qh_lengths[-1]} "
          f"(max_len {args.max_len}).")
    _check_qh_readout_spans(qh_tokenized, tokenizer, n=4)
    target_dist = Counter(r["qh_target_letter"] for r in qh_tokenized)
    weighting = (
        "same binary NLL; grounded+qh-aux defaults to lambda=2 because two grounded "
        "Y_true-supporting views are present"
        if qy_loss == QY_LOSS_FORCED_CHOICE
        else f"token-CE Q_Y; its answer letter is averaged across C={constant_c} tokens"
    )
    print(f"--check-tokenizer [--qh-aux] complete: qy_loss={qy_loss!r}; "
          f"{len(qh_tokenized)} Q_H readout inputs, targets A={target_dist['A']} "
          f"B={target_dist['B']}, lambda={qh_lambda_of(args)} ({weighting}).")


# ---------------------------------------------------------------------------
# bf16 base preparation (cluster, once)
# ---------------------------------------------------------------------------

def _from_pretrained_compat(auto_cls, name_or_path, dtype, **kwargs):
    """transformers renamed torch_dtype= to dtype=; support both."""
    try:
        return auto_cls.from_pretrained(name_or_path, dtype=dtype, **kwargs)
    except TypeError:
        return auto_cls.from_pretrained(name_or_path, torch_dtype=dtype, **kwargs)


def prepare_bf16():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    if os.path.isdir(BF16_DIR):
        raise RuntimeError(
            f"{BF16_DIR} already exists — refusing to overwrite a prepared base. "
            "Delete it manually if you really want to regenerate."
        )
    print(f"Dequantizing {MODEL_NAME} (MXFP4 -> bf16) on CPU; this is slow but one-time ...")
    model = _from_pretrained_compat(
        AutoModelForCausalLM, MODEL_NAME, torch.bfloat16,
        attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True),
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError(
            "quantization_config survived the dequantize load — this transformers "
            "version did not strip it; a quantizer-free bf16 base cannot be saved. "
            "Check the installed transformers version."
        )
    model.save_pretrained(BF16_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(BF16_DIR)

    with open(os.path.join(BF16_DIR, "config.json")) as f:
        saved = json.load(f)
    assert "quantization_config" not in saved, "saved config still has quantization_config"
    saved_dtype = saved.get("dtype") or saved.get("torch_dtype")
    assert saved_dtype == "bfloat16", f"saved base is {saved_dtype!r}, expected bfloat16"
    print(f"Saved quantizer-free bf16 base + tokenizer to {BF16_DIR}.")


# ---------------------------------------------------------------------------
# Full-parameter training preflight
# ---------------------------------------------------------------------------

def _gb(n_bytes):
    return n_bytes / 2 ** 30


def _memory_table(state_bytes_per_param=STATE_BYTES_PER_PARAM, overhead_gb=PER_GPU_OVERHEAD_GB,
                  h100_gb=80):
    lines = [f"    GPUs | sharded state/GPU + overhead | fits {h100_gb}GB H100?"]
    for w in (1, 2, 4, 8):
        need = _gb(state_bytes_per_param * N_PARAMS_EST) / w + overhead_gb
        lines.append(f"    {w:>4} | {need:>7.0f} GB                    | "
                     f"{'yes' if need <= h100_gb * GPU_MEM_SAFETY else 'NO'}")
    return "\n".join(lines)


def preflight(world_size, use_deepspeed):
    import torch
    import transformers

    fsdp_tail = (
        "\nFSDP no-offload full FT keeps fp32 master weights/optimizer state on GPU "
        f"(~{STATE_BYTES_PER_PARAM} bytes/param of sharded state ~ "
        f"{_gb(STATE_BYTES_PER_PARAM * N_PARAMS_EST):.0f} GiB total "
        f"+ ~{PER_GPU_OVERHEAD_GB:.0f} GiB/GPU transients):\n" + _memory_table() +
        "\nThis script performs full-parameter training and never substitutes LoRA. For 4x H100 "
        "96GB + 512GB CPU RAM, use the DeepSpeed optimizer-offload path:\n"
        f"  accelerate launch --config_file {DEEPSPEED_CONFIG_PATH} ft-verifier-oss-full.py"
    )
    ds_tail = (
        "\nDeepSpeed ZeRO-3 optimizer-offload full FT keeps fp32 optimizer/master "
        "state on CPU and bf16 params/grads sharded on GPU "
        f"(GPU estimate: ~{DS_GPU_BYTES_PER_PARAM} bytes/param + "
        f"~{DS_PER_GPU_OVERHEAD_GB:.0f} GiB/GPU; CPU offload estimate: "
        f"~{_gb(DS_CPU_OFFLOAD_BYTES_PER_PARAM * N_PARAMS_EST):.0f} GiB plus "
        "checkpoint/save headroom). There is NO LoRA fallback."
    )
    msg_tail = ds_tail if use_deepspeed else fsdp_tail

    # Version pins (match ft-oss.py guidance; FSDP save behavior is version-sensitive).
    def _vtuple(v):
        return tuple(int(p.split("+")[0]) for p in v.split(".")[:3]
                     if p.split("+")[0].isdigit())

    tf_v = transformers.__version__
    if not (_vtuple("4.56.2") <= _vtuple(tf_v) < _vtuple("5.0.0")):
        raise RuntimeError(f"transformers {tf_v} unsupported; pin >=4.56.2,<5." + msg_tail)
    torch_v = torch.__version__
    if _vtuple(torch_v) < _vtuple("2.4.0"):
        raise RuntimeError(f"torch {torch_v} is unsupported; need >= 2.4." + msg_tail)
    import accelerate  # noqa: F401  (required by Trainer's FSDP/DeepSpeed paths)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available on this machine." + msg_tail)

    if use_deepspeed:
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
            raise RuntimeError(
                "DeepSpeed path requires launching through accelerate:\n"
                f"  accelerate launch --config_file {DEEPSPEED_CONFIG_PATH} "
                "ft-verifier-oss-full.py" + msg_tail
            )
        config_path = os.environ.get("ACCELERATE_CONFIG_FILE", DEEPSPEED_CONFIG_PATH)
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_text = f.read()
        elif os.path.exists(DEEPSPEED_CONFIG_PATH):
            with open(DEEPSPEED_CONFIG_PATH) as f:
                config_text = f.read()
        else:
            raise RuntimeError(f"DeepSpeed config {DEEPSPEED_CONFIG_PATH} not found." + msg_tail)
        if "offload_optimizer_device: cpu" not in config_text:
            raise RuntimeError(
                "DeepSpeed is enabled, but optimizer CPU offload is not configured. "
                f"Use/update {DEEPSPEED_CONFIG_PATH} with offload_optimizer_device: cpu."
                + msg_tail
            )
    elif os.environ.get("LOCAL_RANK") is None:
        raise RuntimeError("FSDP path requires a torchrun launch (LOCAL_RANK unset)." + msg_tail)

    if 8 % world_size != 0:
        raise RuntimeError(
            f"world_size={world_size} does not divide the target effective batch 8 — "
            "grad accumulation would be non-integer and the effective batch/LR would "
            "silently change. Use 8 GPUs (or a divisor of 8 with enough VRAM)." + msg_tail
        )

    n_gpu = torch.cuda.device_count()
    if use_deepspeed:
        per_gpu_need = _gb(DS_GPU_BYTES_PER_PARAM * N_PARAMS_EST) / world_size + DS_PER_GPU_OVERHEAD_GB
        cpu_need_gb = max(DS_MIN_CPU_RAM_GB, _gb(DS_CPU_OFFLOAD_BYTES_PER_PARAM * N_PARAMS_EST) + 160.0)
    else:
        per_gpu_need = _gb(STATE_BYTES_PER_PARAM * N_PARAMS_EST) / world_size + PER_GPU_OVERHEAD_GB
        cpu_need_gb = MIN_CPU_RAM_GB
    min_vram = min(_gb(torch.cuda.get_device_properties(i).total_memory) for i in range(n_gpu))
    if per_gpu_need > GPU_MEM_SAFETY * min_vram:
        raise RuntimeError(
            f"Estimated {per_gpu_need:.0f} GB/GPU at world_size={world_size} exceeds "
            f"{GPU_MEM_SAFETY:.0%} of the smallest visible GPU ({min_vram:.0f} GB)." + msg_tail
        )

    try:
        with open("/proc/meminfo") as f:
            available_kb = next(int(line.split()[1]) for line in f
                                if line.startswith("MemAvailable:"))
        if available_kb / 2 ** 20 < cpu_need_gb:
            raise RuntimeError(
                f"Only {available_kb / 2 ** 20:.0f} GB CPU RAM available; need ~"
                f"{cpu_need_gb:.0f} GB "
                f"({'ZeRO-3 optimizer offload + save headroom' if use_deepspeed else 'rank0 fp32 load + FULL_STATE_DICT gather'})."
                + msg_tail
            )
    except (FileNotFoundError, StopIteration):
        print("WARNING: could not read /proc/meminfo; skipping CPU RAM preflight.")

    # Both paths train from the prepared quantizer-free bf16 base: loading the
    # MXFP4 hub repo directly would either re-enter the quantizer (FSDP: defeats
    # rank0-only CPU loading; ZeRO-3: blocks zero3 init) or fail outright.
    if not os.path.isdir(BF16_DIR) or not os.path.exists(os.path.join(BF16_DIR, "config.json")):
        raise RuntimeError(
            f"{BF16_DIR} missing — run `python3 ft-verifier-oss-full.py --prepare-bf16` "
            "first (one-time MXFP4 -> bf16 dequantize)." + msg_tail
        )
    with open(os.path.join(BF16_DIR, "config.json")) as f:
        if "quantization_config" in json.load(f):
            raise RuntimeError(f"{BF16_DIR} still carries quantization_config — "
                               "regenerate it with --prepare-bf16." + msg_tail)

    names = sorted({torch.cuda.get_device_properties(i).name for i in range(n_gpu)})
    print(f"Preflight OK: torch {torch_v}, transformers {tf_v}, {n_gpu} GPU(s) visible "
          f"({', '.join(names)}), world_size={world_size}, "
          f"~{per_gpu_need:.0f} GB/GPU planned, path={'deepspeed-zero3' if use_deepspeed else 'fsdp'}.")


def patch_multiprocess_resource_tracker_shutdown():
    """Suppress a Python 3.12/multiprocess shutdown-only ResourceTracker bug."""
    try:
        import multiprocess.resource_tracker as resource_tracker
    except Exception:
        return

    tracker_cls = getattr(resource_tracker, "ResourceTracker", None)
    if tracker_cls is None or getattr(tracker_cls, "_aisi_shutdown_patch", False):
        return

    original_del = tracker_cls.__del__

    def safe_del(self):
        try:
            return original_del(self)
        except AttributeError as exc:
            if "_recursion_count" in str(exc):
                return None
            raise

    tracker_cls.__del__ = safe_del
    tracker_cls._aisi_shutdown_patch = True


# ---------------------------------------------------------------------------
# Q_H auxiliary loss plumbing (collator / running stats / Trainer subclass)
# ---------------------------------------------------------------------------

def resolve_logits_to_keep_kwarg(model):
    """Which 'only materialize the last logits' kwarg this model's forward accepts.

    transformers renamed num_logits_to_keep -> logits_to_keep. The name must be
    resolved from the UNWRAPPED model's signature: a DeepSpeedEngine / FSDP wrapper
    forwards **kwargs blindly, and gpt-oss' forward also swallows unknown kwargs via
    **kwargs: Unpack[TransformersKwargs] — so guessing would silently compute full
    logits (or explode deep inside). Returns None => full-logits fallback.
    """
    import inspect

    target = getattr(model, "forward", model)
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        param = params.get(name)
        if param is not None and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            return name
    return None


def qh_reference_nll(z_a, z_b, target_index):
    """Restricted two-letter NLL in stdlib float64: -(z_target - logsumexp(z_A, z_B)).

    Used as an offline-testable reference AND as a --smoke cross-check of the torch
    implementation, so the exact loss formula is verifiable without a GPU.
    """
    if target_index not in (0, 1):
        raise ValueError(f"target_index must be 0 (A) or 1 (B), got {target_index!r}")
    z = (float(z_a), float(z_b))
    top = max(z)
    logsumexp = top + math.log(math.exp(z[0] - top) + math.exp(z[1] - top))
    return logsumexp - z[target_index]


def forced_choice_leg(model, input_ids, attention_mask, target, label, letter_id_a,
                      letter_id_b, logits_kwarg, prefill_last_id):
    """ONE restricted last-position A/B NLL. The single implementation shared by the
    --qh-aux arm (Q_Y and Q_H legs) and the grounded arm's direct view, so the three
    readouts cannot drift apart. No labels are ever passed to the model."""
    import torch

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f"{label} forced-choice input must have shape [1, T], got "
            f"{tuple(input_ids.shape)}"
        )
    if int(input_ids[0, -1].detach()) != int(prefill_last_id):
        raise RuntimeError(
            f"{label} forced-choice tensor does not end at FINAL_CHANNEL_PREFILL; "
            "refusing to score logits from the wrong input key or position"
        )
    target = target.view(-1)
    if target.numel() != 1 or int(target[0].detach()) not in (0, 1):
        raise RuntimeError(
            f"{label} forced-choice target must be one 0/1 index, got "
            f"{target.detach().tolist()}"
        )
    forward_kwargs = {logits_kwarg: 1} if logits_kwarg else {}
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **forward_kwargs,
    )
    logits = outputs.logits
    if logits.dim() != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"unexpected {label} logits shape {tuple(logits.shape)}")
    if logits.shape[1] not in (1, input_ids.shape[1]):
        raise RuntimeError(
            f"{label} logits have {logits.shape[1]} positions for a "
            f"{input_ids.shape[1]}-token input; neither the last-position slice nor "
            "the full-logits fallback. Refusing to score an unknown position."
        )
    last = logits[0, -1, :]
    # Cast the selected A/B logits to fp32 BEFORE logsumexp. This is the exact
    # numerical path for every leg, so lambdas weight like quantities.
    z = torch.stack([last[int(letter_id_a)], last[int(letter_id_b)]]).float()
    log_p = z - torch.logsumexp(z, dim=-1)
    loss = -log_p.gather(0, target).squeeze(0)
    return outputs, loss, log_p, z, target


class QHRunningStats:
    """Rank-local running means between two Trainer.log calls.

    Deliberately plain floats: no tensors are kept alive and no collective is added,
    so logging cannot perturb gradients, memory or the distributed schedule.
    """

    def __init__(self):
        self._sums = {}
        self._n = 0

    def add(self, **values):
        for key, value in values.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._n += 1

    @property
    def count(self):
        return self._n

    def pop_means(self, ndigits=6):
        if not self._n:
            return {}
        means = {k: round(v / self._n, ndigits) for k, v in self._sums.items()}
        self._sums, self._n = {}, 0
        return means


class QHPairCollator:
    """Right-pad both legs of a paired ``(Q_Y, Q_H)`` microbatch.

    Token-CE Q_Y uses ``PadCollator``. In forced-choice mode, the Q_Y readout ends
    immediately before its answer letter; labels remain only for Trainer's labelled-batch
    evaluation routing. Both forced-choice readouts score their final position, which
    requires microbatch size one.
    """

    def __init__(self, pad_token_id, qy_loss=QY_LOSS_FORCED_CHOICE,
                 letter_ids=None, prefill_ids=None, return_token_id=None):
        self.pad_token_id = pad_token_id
        self.qy_collator = PadCollator(pad_token_id)
        self.qy_loss = qy_loss
        if qy_loss not in QY_LOSS_MODES:
            raise RuntimeError(f"unknown Q_Y loss for paired collator: {qy_loss!r}")
        self.letter_ids = ({k: int(v) for k, v in letter_ids.items()}
                           if letter_ids else None)
        self.prefill_ids = [int(i) for i in prefill_ids] if prefill_ids else None
        self.return_token_id = (None if return_token_id is None
                                else int(return_token_id))
        if self.qy_loss == QY_LOSS_FORCED_CHOICE:
            if not self.letter_ids or set(self.letter_ids) != {"A", "B"}:
                raise RuntimeError("forced-choice Q_Y collator requires A/B letter ids")
            if not self.prefill_ids or self.return_token_id is None:
                raise RuntimeError(
                    "forced-choice Q_Y collator requires prefill and return-token ids"
                )

    def _validate_qy_forced_choice_feature(self, feature):
        full_ids = [int(i) for i in feature["input_ids"]]
        labels = [int(i) for i in feature["labels"]]
        readout_ids = [int(i) for i in feature["qy_readout_input_ids"]]
        readout_mask = [int(i) for i in feature["qy_readout_attention_mask"]]
        target = int(feature["qy_target_index"])
        if target not in (0, 1):
            raise RuntimeError(f"Q_Y forced-choice target must be 0/1, got {target!r}")
        target_letter = "A" if target == 0 else "B"
        if len(full_ids) < QY_READOUT_TAIL_TOKENS:
            raise RuntimeError("Q_Y tokenized row is too short to contain letter + return")
        if readout_ids != full_ids[:-QY_READOUT_TAIL_TOKENS]:
            raise RuntimeError(
                "Q_Y forced-choice readout is not exactly the audited input with its "
                "target letter and return token removed"
            )
        if readout_mask != [1] * len(readout_ids):
            raise RuntimeError("Q_Y forced-choice readout attention mask is not all ones")
        if readout_ids[-len(self.prefill_ids):] != self.prefill_ids:
            raise RuntimeError("Q_Y forced-choice readout does not end at the exact prefill")
        if full_ids[-2] != self.letter_ids[target_letter]:
            raise RuntimeError(
                "Q_Y forced-choice target index disagrees with the audited answer token"
            )
        if full_ids[-1] != self.return_token_id:
            raise RuntimeError("Q_Y audited input does not end with the canonical return token")
        if labels[-2] != full_ids[-2] or labels[-1] != full_ids[-1]:
            raise RuntimeError(
                "Q_Y labels do not expose the same answer letter + return as the audited row"
            )

    def __call__(self, features):
        import torch

        if len(features) != 1:
            raise RuntimeError(
                "the Q_H auxiliary arm requires exactly one paired example per microbatch "
                f"(per_device batch size 1), got {len(features)}: the last-position readout "
                "would otherwise land on padding."
            )
        if self.qy_loss == QY_LOSS_FORCED_CHOICE:
            for feature in features:
                self._validate_qy_forced_choice_feature(feature)
        batch = self.qy_collator(features)
        if self.qy_loss == QY_LOSS_FORCED_CHOICE:
            max_qy_len = max(len(f["qy_readout_input_ids"]) for f in features)
            qy_input_ids, qy_attention_mask = [], []
            for feature in features:
                pad = max_qy_len - len(feature["qy_readout_input_ids"])
                qy_input_ids.append(
                    list(feature["qy_readout_input_ids"]) + [self.pad_token_id] * pad
                )
                qy_attention_mask.append(
                    list(feature["qy_readout_attention_mask"]) + [0] * pad
                )
            batch["qy_readout_input_ids"] = torch.tensor(qy_input_ids, dtype=torch.long)
            batch["qy_readout_attention_mask"] = torch.tensor(
                qy_attention_mask, dtype=torch.long
            )
            batch["qy_target_index"] = torch.tensor(
                [int(f["qy_target_index"]) for f in features], dtype=torch.long
            )
        max_len = max(len(f["qh_input_ids"]) for f in features)
        qh_input_ids, qh_attention_mask = [], []
        for f in features:
            pad = max_len - len(f["qh_input_ids"])
            qh_input_ids.append(list(f["qh_input_ids"]) + [self.pad_token_id] * pad)
            qh_attention_mask.append(list(f["qh_attention_mask"]) + [0] * pad)
        batch["qh_input_ids"] = torch.tensor(qh_input_ids, dtype=torch.long)
        batch["qh_attention_mask"] = torch.tensor(qh_attention_mask, dtype=torch.long)
        batch["qh_target_index"] = torch.tensor(
            [int(f["qh_target_index"]) for f in features], dtype=torch.long
        )
        return batch


def qh_aux_trainer_class(Trainer):
    """Build the QHAuxTrainer subclass (Trainer is imported lazily inside train())."""

    class QHAuxTrainer(Trainer):
        """Two forwards (Q_Y then Q_H), one backward on their weighted sum.

        ``model_accepts_loss_kwargs`` is false so Trainer does not inject
        ``num_items_in_batch``. The default mode applies one shared restricted-A/B NLL
        implementation to both readouts; ``token-ce`` selects the alternate supervised-
        continuation Q_Y objective.
        """

        def __init__(self, *args, qh_lambda=QH_AUX_DEFAULT_LAMBDA, qh_letter_ids=None,
                     qh_logits_kwarg=None, qh_smoke=False,
                     qy_loss=QY_LOSS_FORCED_CHOICE,
                     readout_prefill_last_id=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if not qh_letter_ids:
                raise RuntimeError("QHAuxTrainer requires the ' A'/' B' token ids")
            if qy_loss not in QY_LOSS_MODES:
                raise RuntimeError(f"QHAuxTrainer got unknown Q_Y loss {qy_loss!r}")
            self.qh_lambda = float(qh_lambda)
            self.qy_loss = qy_loss
            self.qh_letter_id_a = int(qh_letter_ids["A"])
            self.qh_letter_id_b = int(qh_letter_ids["B"])
            self.qh_logits_kwarg = qh_logits_kwarg
            if readout_prefill_last_id is None:
                raise RuntimeError("QHAuxTrainer requires the final readout-prefill token id")
            self.readout_prefill_last_id = int(readout_prefill_last_id)
            self.qh_smoke = bool(qh_smoke)
            self.qh_stats = QHRunningStats()
            self.qy_forwards = 0
            self.qh_forwards = 0
            self.qy_logits_shape_seen = None
            self.qh_logits_shape_seen = None

        def _forced_choice_leg(self, model, input_ids, attention_mask, target, label):
            """One shared last-position A/B NLL for both Q_Y and Q_H.

            Delegates to the module-level ``forced_choice_leg`` shared with the grounded
            direct view."""
            return forced_choice_leg(
                model, input_ids, attention_mask, target, label,
                self.qh_letter_id_a, self.qh_letter_id_b, self.qh_logits_kwarg,
                self.readout_prefill_last_id,
            )

        def _check_reference_nll(self, loss, z, target, label):
            stats = [
                float(loss.detach()), float(z.detach()[0]), float(z.detach()[1]),
                int(target.detach()[0]),
            ]
            reference = qh_reference_nll(stats[1], stats[2], stats[3])
            if abs(reference - stats[0]) > 1e-4:
                raise RuntimeError(
                    f"{label} loss {stats[0]!r} disagrees with the stdlib reference "
                    f"{reference!r} for z=({stats[1]}, {stats[2]}); the restricted "
                    "two-letter NLL is not being computed as documented."
                )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,
                         **kwargs):
            import torch

            if num_items_in_batch is not None:
                raise RuntimeError(
                    "num_items_in_batch reached the paired Q_Y/Q_H loss; "
                    "model_accepts_loss_kwargs must stay False."
                )
            if self.model_accepts_loss_kwargs:
                raise RuntimeError("model_accepts_loss_kwargs was re-enabled under --qh-aux")
            input_ids = inputs["input_ids"]
            is_training = bool(getattr(model, "training", True))
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    f"Q_H auxiliary loss expects microbatch size 1, got {input_ids.shape[0]}"
                )

            qy_log_p = qy_z = qy_target = None
            if self.qy_loss == QY_LOSS_FORCED_CHOICE:
                # The original labels intentionally remain in `inputs` for Trainer's
                # evaluation routing, but they are NOT passed to the model on this branch.
                outputs_y, loss_y, qy_log_p, qy_z, qy_target = self._forced_choice_leg(
                    model,
                    inputs["qy_readout_input_ids"],
                    inputs["qy_readout_attention_mask"],
                    inputs["qy_target_index"],
                    "Q_Y",
                )
                self.qy_logits_shape_seen = int(outputs_y.logits.shape[1])
            else:
                # Alternate token-CE Q_Y objective over the audited continuation.
                outputs_y = model(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                loss_y = outputs_y.loss
            if is_training:
                self.qy_forwards += 1
            if self.qh_smoke and qy_z is not None:
                self._check_reference_nll(loss_y, qy_z, qy_target, "Q_Y")
            if self.qh_lambda == 0.0:
                # Smoke-only plumbing check: the Q_H forward is skipped EXACTLY.
                values = {"loss_qy": float(loss_y.detach())}
                if qy_log_p is not None:
                    values.update(
                        qy_p_target=float(
                            qy_log_p.detach().gather(0, qy_target).squeeze(0).exp()
                        ),
                        qy_acc=float(
                            qy_log_p.detach().argmax(0) == qy_target.squeeze(0)
                        ),
                    )
                if is_training:
                    self.qh_stats.add(**values)
                return (loss_y, outputs_y) if return_outputs else loss_y

            # Leg 2: the Q_H forced-choice readout. No labels => no second LM loss.
            qh_input_ids = inputs["qh_input_ids"]
            outputs_h, loss_h, qh_log_p, qh_z, qh_target = self._forced_choice_leg(
                model,
                qh_input_ids,
                inputs["qh_attention_mask"],
                inputs["qh_target_index"],
                "Q_H",
            )
            if is_training:
                self.qh_forwards += 1
            self.qh_logits_shape_seen = int(outputs_h.logits.shape[1])
            loss = loss_y + self.qh_lambda * loss_h

            with torch.no_grad():
                values = {
                    "loss_qy": float(loss_y.detach()),
                    "loss_qh": float(loss_h.detach()),
                    "qh_p_target": float(
                        qh_log_p.detach().gather(0, qh_target).squeeze(0).exp()
                    ),
                    "qh_acc": float(
                        qh_log_p.detach().argmax(0) == qh_target.squeeze(0)
                    ),
                }
                if qy_log_p is not None:
                    values.update(
                        qy_p_target=float(
                            qy_log_p.detach().gather(0, qy_target).squeeze(0).exp()
                        ),
                        qy_acc=float(
                            qy_log_p.detach().argmax(0) == qy_target.squeeze(0)
                        ),
                    )
            if is_training:
                self.qh_stats.add(**values)
            if self.qh_smoke:
                self._check_reference_nll(loss_h, qh_z, qh_target, "Q_H")
            return (loss, outputs_y) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            means = self.qh_stats.pop_means()
            if means:
                logs = {
                    **logs, **means,
                    "qy_forwards": self.qy_forwards,
                    "qh_forwards": self.qh_forwards,
                }
            return super().log(logs, *args, **kwargs)

    return QHAuxTrainer


# ---------------------------------------------------------------------------
# Grounded arm loss plumbing (collator / Trainer subclass)
# ---------------------------------------------------------------------------

class GroundedQHPairCollator(QHPairCollator):
    """Collate the three-view grounded+Q_H positive-control microbatch.

    QHPairCollator pads the direct Q_Y and Q_H readouts and validates the Q_H plumbing.
    Its answer-only Q_Y validator cannot be reused as-is: a resolved grounded LM row
    contains analysis tokens which are deliberately absent from the direct Q_Y view.
    Override only that validator so the pure answer-only Q_H collator remains untouched.
    """

    def _validate_qy_forced_choice_feature(self, feature):
        # GroundedCollator is defined below and is available by the time any collator is
        # instantiated. Its validator accepts both resolved analysis rows and unresolved
        # answer-only fallbacks while enforcing the shared prompt/prefill/answer slot.
        GroundedCollator._validate(self, feature)

    def __call__(self, features):
        import torch

        batch = super().__call__(features)
        batch["grounded_lm_enabled"] = torch.tensor(
            [bool(feature.get("grounded_lm_enabled", True)) for feature in features],
            dtype=torch.bool,
        )
        return batch


def grounded_qh_aux_trainer_class(Trainer):
    """Build the combined grounded + Q_H positive-control Trainer.

    The combined arm is deliberately its own class rather than a precedence trick
    between ``GroundedTrainer`` and ``QHAuxTrainer``. It performs up to three forwards
    on one microbatch and one backward on their weighted sum:

      * grounded LM analysis view (skipped for unresolved answer-only fallbacks),
      * grounded direct Q_Y forced-choice view, and
      * the directly supervised Q_H forced-choice positive-control view.

    ``model_accepts_loss_kwargs`` stays False so Trainer cannot apply token-count
    normalization to only one leg. Every forced-choice leg uses the module-level shared
    ``forced_choice_leg`` implementation.
    """

    class GroundedQHAuxTrainer(Trainer):
        def __init__(self, *args, lambda_lm=GROUNDED_LAMBDA_LM_DEFAULT,
                     lambda_fc=GROUNDED_LAMBDA_FC_DEFAULT,
                     qh_lambda=QH_AUX_DEFAULT_LAMBDA, letter_ids=None,
                     logits_kwarg=None, smoke=False,
                     readout_prefill_last_id=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if not letter_ids or set(letter_ids) != {"A", "B"}:
                raise RuntimeError("GroundedQHAuxTrainer requires the ' A'/' B' token ids")
            if readout_prefill_last_id is None:
                raise RuntimeError(
                    "GroundedQHAuxTrainer requires the final readout-prefill token id"
                )
            self.lambda_lm = float(lambda_lm)
            self.lambda_fc = float(lambda_fc)
            self.qh_lambda = float(qh_lambda)
            self.letter_id_a = int(letter_ids["A"])
            self.letter_id_b = int(letter_ids["B"])
            self.logits_kwarg = logits_kwarg
            self.readout_prefill_last_id = int(readout_prefill_last_id)
            self.smoke = bool(smoke)
            self.stats = QHRunningStats()
            self.lm_forwards = 0
            self.fc_forwards = 0  # grounded direct Q_Y view; kept for metadata parity
            self.qy_forwards = 0   # explicit alias used by Q_H-arm diagnostics
            self.qh_forwards = 0
            self.lm_logits_shape_seen = None
            self.fc_logits_shape_seen = None
            # The shared Q_H metadata path calls the direct grounded view Q_Y.
            # Keep the explicit ``fc`` name for grounded-only metadata, but expose
            # the Q_Y alias as well when this three-view trainer is used.
            self.qy_logits_shape_seen = None
            self.qh_logits_shape_seen = None

        def _check_reference_nll(self, loss, z, target, label):
            stats = [
                float(loss.detach()), float(z.detach()[0]), float(z.detach()[1]),
                int(target.detach()[0]),
            ]
            reference = qh_reference_nll(stats[1], stats[2], stats[3])
            if abs(reference - stats[0]) > 1e-4:
                raise RuntimeError(
                    f"{label} loss {stats[0]!r} disagrees with the stdlib reference "
                    f"{reference!r} for z=({stats[1]}, {stats[2]})"
                )

        def _forced(self, model, input_ids, attention_mask, target, label):
            return forced_choice_leg(
                model, input_ids, attention_mask, target, label,
                self.letter_id_a, self.letter_id_b, self.logits_kwarg,
                self.readout_prefill_last_id,
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,
                         **kwargs):
            import torch

            if num_items_in_batch is not None:
                raise RuntimeError(
                    "num_items_in_batch reached the grounded+Q_H loss; "
                    "model_accepts_loss_kwargs must stay False."
                )
            if self.model_accepts_loss_kwargs:
                raise RuntimeError(
                    "model_accepts_loss_kwargs was re-enabled under grounded+qh-aux"
                )
            input_ids = inputs["input_ids"]
            is_training = bool(getattr(model, "training", True))
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    f"the grounded+Q_H loss expects microbatch size 1, got {input_ids.shape[0]}"
                )
            lm_flag = inputs.get("grounded_lm_enabled")
            lm_enabled = True if lm_flag is None else bool(lm_flag.view(-1)[0])

            outputs_lm, loss_lm = None, None
            if lm_enabled and self.lambda_lm != 0.0:
                outputs_lm = model(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                loss_lm = outputs_lm.loss
                self.lm_logits_shape_seen = int(outputs_lm.logits.shape[1])
                if is_training:
                    self.lm_forwards += 1

            outputs_qy = loss_qy = qy_log_p = qy_z = qy_target = None
            if self.lambda_fc != 0.0:
                outputs_qy, loss_qy, qy_log_p, qy_z, qy_target = self._forced(
                    model,
                    inputs["qy_readout_input_ids"],
                    inputs["qy_readout_attention_mask"],
                    inputs["qy_target_index"],
                    "grounded Q_Y",
                )
                self.fc_logits_shape_seen = int(outputs_qy.logits.shape[1])
                self.qy_logits_shape_seen = self.fc_logits_shape_seen
                if is_training:
                    self.fc_forwards += 1
                    self.qy_forwards += 1
                if self.smoke:
                    self._check_reference_nll(loss_qy, qy_z, qy_target, "grounded Q_Y")

            outputs_qh = loss_qh = qh_log_p = qh_z = qh_target = None
            if self.qh_lambda != 0.0:
                outputs_qh, loss_qh, qh_log_p, qh_z, qh_target = self._forced(
                    model,
                    inputs["qh_input_ids"],
                    inputs["qh_attention_mask"],
                    inputs["qh_target_index"],
                    "Q_H",
                )
                self.qh_logits_shape_seen = int(outputs_qh.logits.shape[1])
                if is_training:
                    self.qh_forwards += 1
                if self.smoke:
                    self._check_reference_nll(loss_qh, qh_z, qh_target, "Q_H")

            losses = [
                self.lambda_lm * loss_lm if loss_lm is not None else None,
                self.lambda_fc * loss_qy if loss_qy is not None else None,
                self.qh_lambda * loss_qh if loss_qh is not None else None,
            ]
            losses = [value for value in losses if value is not None]
            if not losses:
                raise RuntimeError(
                    "grounded+qh-aux has no active loss leg; at least one non-zero "
                    "lambda is required"
                )
            loss = sum(losses[1:], losses[0])

            with torch.no_grad():
                values = {
                    "lm_view_used": float(loss_lm is not None),
                    "qy_view_used": float(loss_qy is not None),
                    "qh_view_used": float(loss_qh is not None),
                }
                if loss_lm is not None:
                    values["loss_lm"] = float(loss_lm.detach())
                if loss_qy is not None:
                    values.update(
                        loss_qy=float(loss_qy.detach()),
                        qy_p_target=float(
                            qy_log_p.detach().gather(0, qy_target).squeeze(0).exp()
                        ),
                        qy_acc=float(
                            qy_log_p.detach().argmax(0) == qy_target.squeeze(0)
                        ),
                    )
                if loss_qh is not None:
                    values.update(
                        loss_qh=float(loss_qh.detach()),
                        qh_p_target=float(
                            qh_log_p.detach().gather(0, qh_target).squeeze(0).exp()
                        ),
                        qh_acc=float(
                            qh_log_p.detach().argmax(0) == qh_target.squeeze(0)
                        ),
                    )
            if is_training:
                self.stats.add(**values)
            outputs = outputs_lm
            if outputs is None:
                outputs = outputs_qy
            if outputs is None:
                outputs = outputs_qh
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            means = self.stats.pop_means()
            if means:
                logs = {
                    **logs, **means,
                    "lm_forwards": self.lm_forwards,
                    "qy_forwards": self.qy_forwards,
                    "qh_forwards": self.qh_forwards,
                }
            return super().log(logs, *args, **kwargs)

    return GroundedQHAuxTrainer

class GroundedCollator:
    """Right-pad the grounded LM view and attach its DIRECT forced-choice view.

    The LM view is delegated byte-for-byte to PadCollator. The direct view is the same
    chat prompt followed by FINAL_CHANNEL_PREFILL — no analysis in context — and is scored
    at its LAST position, hence the hard batch-size-1 requirement. Every structural
    invariant is checked here, before any tensor exists, so a drifted readout fails closed
    instead of silently training on the wrong slot.
    """

    def __init__(self, pad_token_id, letter_ids=None, prefill_ids=None,
                 return_token_id=None):
        self.pad_token_id = pad_token_id
        self.lm_collator = PadCollator(pad_token_id)
        self.letter_ids = {k: int(v) for k, v in (letter_ids or {}).items()}
        if set(self.letter_ids) != {"A", "B"}:
            raise RuntimeError("grounded collator requires A/B letter ids")
        if not prefill_ids or return_token_id is None:
            raise RuntimeError("grounded collator requires prefill and return-token ids")
        self.prefill_ids = [int(i) for i in prefill_ids]
        self.return_token_id = int(return_token_id)

    def _validate(self, feature):
        full_ids = [int(i) for i in feature["input_ids"]]
        labels = [int(i) for i in feature["labels"]]
        readout_ids = [int(i) for i in feature["qy_readout_input_ids"]]
        readout_mask = [int(i) for i in feature["qy_readout_attention_mask"]]
        target = int(feature["qy_target_index"])
        if target not in (0, 1):
            raise RuntimeError(f"grounded forced-choice target must be 0/1, got {target!r}")
        target_letter = "A" if target == 0 else "B"
        if len(full_ids) < QY_READOUT_TAIL_TOKENS + len(self.prefill_ids):
            raise RuntimeError("grounded tokenized row is too short to hold its answer slot")
        if readout_mask != [1] * len(readout_ids):
            raise RuntimeError("grounded direct readout attention mask is not all ones")
        if readout_ids[-len(self.prefill_ids):] != self.prefill_ids:
            raise RuntimeError("grounded direct readout does not end at the exact prefill")
        prompt_len = len(readout_ids) - len(self.prefill_ids)
        if readout_ids[:prompt_len] != full_ids[:prompt_len]:
            raise RuntimeError(
                "grounded direct readout prompt prefix differs from the LM view's prompt"
            )
        if full_ids[-2] != self.letter_ids[target_letter]:
            raise RuntimeError(
                "grounded forced-choice target index disagrees with the audited answer token"
            )
        if full_ids[-1] != self.return_token_id:
            raise RuntimeError(
                "grounded audited input does not end with the canonical return token"
            )
        if labels[-2] != full_ids[-2] or labels[-1] != full_ids[-1]:
            raise RuntimeError(
                "grounded labels do not expose the same answer letter + return as the row"
            )

    def __call__(self, features):
        import torch

        if len(features) != 1:
            raise RuntimeError(
                "the grounded arm requires exactly one example per microbatch "
                f"(per_device batch size 1), got {len(features)}: the last-position "
                "readout would otherwise land on padding."
            )
        for feature in features:
            self._validate(feature)
        batch = self.lm_collator(features)
        max_len = max(len(f["qy_readout_input_ids"]) for f in features)
        readout_ids, readout_mask = [], []
        for feature in features:
            pad = max_len - len(feature["qy_readout_input_ids"])
            readout_ids.append(list(feature["qy_readout_input_ids"]) + [self.pad_token_id] * pad)
            readout_mask.append(list(feature["qy_readout_attention_mask"]) + [0] * pad)
        batch["qy_readout_input_ids"] = torch.tensor(readout_ids, dtype=torch.long)
        batch["qy_readout_attention_mask"] = torch.tensor(readout_mask, dtype=torch.long)
        batch["qy_target_index"] = torch.tensor(
            [int(f["qy_target_index"]) for f in features], dtype=torch.long
        )
        batch["grounded_lm_enabled"] = torch.tensor(
            [bool(f.get("grounded_lm_enabled", True)) for f in features], dtype=torch.bool
        )
        return batch


def grounded_trainer_class(Trainer):
    """Build the GroundedTrainer subclass (Trainer is imported lazily inside train())."""

    class GroundedTrainer(Trainer):
        """Two forwards (grounded LM view, then the direct forced-choice view), one
        backward on their weighted sum.

        model_accepts_loss_kwargs is forced False so the Trainer never injects
        num_items_in_batch: the LM view must stay a per-example token mean and Trainer
        must apply gradient-accumulation normalization to the weighted SUM.
        """

        def __init__(self, *args, lambda_lm=GROUNDED_LAMBDA_LM_DEFAULT,
                     lambda_fc=GROUNDED_LAMBDA_FC_DEFAULT, grounded_letter_ids=None,
                     grounded_logits_kwarg=None, grounded_smoke=False,
                     readout_prefill_last_id=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if not grounded_letter_ids:
                raise RuntimeError("GroundedTrainer requires the ' A'/' B' token ids")
            if readout_prefill_last_id is None:
                raise RuntimeError("GroundedTrainer requires the final readout-prefill id")
            self.lambda_lm = float(lambda_lm)
            self.lambda_fc = float(lambda_fc)
            self.letter_id_a = int(grounded_letter_ids["A"])
            self.letter_id_b = int(grounded_letter_ids["B"])
            self.grounded_logits_kwarg = grounded_logits_kwarg
            self.readout_prefill_last_id = int(readout_prefill_last_id)
            self.grounded_smoke = bool(grounded_smoke)
            self.grounded_stats = QHRunningStats()
            self.lm_forwards = 0
            self.fc_forwards = 0
            self.lm_logits_shape_seen = None
            self.fc_logits_shape_seen = None

        def _check_reference_nll(self, loss, z, target, label):
            stats = [
                float(loss.detach()), float(z.detach()[0]), float(z.detach()[1]),
                int(target.detach()[0]),
            ]
            reference = qh_reference_nll(stats[1], stats[2], stats[3])
            if abs(reference - stats[0]) > 1e-4:
                raise RuntimeError(
                    f"{label} loss {stats[0]!r} disagrees with the stdlib reference "
                    f"{reference!r} for z=({stats[1]}, {stats[2]})"
                )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,
                         **kwargs):
            import torch

            if num_items_in_batch is not None:
                raise RuntimeError(
                    "num_items_in_batch reached the grounded loss; "
                    "model_accepts_loss_kwargs must stay False."
                )
            if self.model_accepts_loss_kwargs:
                raise RuntimeError(
                    "model_accepts_loss_kwargs was re-enabled under --supervision grounded"
                )
            input_ids = inputs["input_ids"]
            is_training = bool(getattr(model, "training", True))
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    f"the grounded loss expects microbatch size 1, got {input_ids.shape[0]}"
                )
            lm_flag = inputs.get("grounded_lm_enabled")
            lm_enabled = True if lm_flag is None else bool(lm_flag.view(-1)[0])

            outputs_lm, loss_lm = None, None
            if lm_enabled and self.lambda_lm != 0.0:
                outputs_lm = model(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                loss_lm = outputs_lm.loss
                self.lm_logits_shape_seen = int(outputs_lm.logits.shape[1])
                if is_training:
                    self.lm_forwards += 1

            outputs_fc, loss_fc, log_p, z, target = forced_choice_leg(
                model,
                inputs["qy_readout_input_ids"],
                inputs["qy_readout_attention_mask"],
                inputs["qy_target_index"],
                "grounded Q_Y",
                self.letter_id_a,
                self.letter_id_b,
                self.grounded_logits_kwarg,
                self.readout_prefill_last_id,
            )
            self.fc_logits_shape_seen = int(outputs_fc.logits.shape[1])
            if is_training:
                self.fc_forwards += 1
            if self.grounded_smoke:
                self._check_reference_nll(loss_fc, z, target, "grounded Q_Y")

            loss = self.lambda_fc * loss_fc
            if loss_lm is not None:
                loss = self.lambda_lm * loss_lm + loss

            with torch.no_grad():
                values = {
                    "loss_fc": float(loss_fc.detach()),
                    "fc_p_target": float(
                        log_p.detach().gather(0, target).squeeze(0).exp()
                    ),
                    "fc_acc": float(log_p.detach().argmax(0) == target.squeeze(0)),
                    "lm_view_used": float(loss_lm is not None),
                }
                if loss_lm is not None:
                    values["loss_lm"] = float(loss_lm.detach())
            if is_training:
                self.grounded_stats.add(**values)
            outputs = outputs_lm if outputs_lm is not None else outputs_fc
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            means = self.grounded_stats.pop_means()
            if means:
                logs = {
                    **logs, **means,
                    "lm_forwards": self.lm_forwards,
                    "fc_forwards": self.fc_forwards,
                }
            return super().log(logs, *args, **kwargs)

    return GroundedTrainer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def effective_output_dir(args):
    """Checkpoint directory after applying the existing smoke suffix policy."""
    output_dir = args.output_dir
    if args.smoke:
        output_dir = output_dir.rstrip("/") + "-smoke"
    return output_dir


def guard_checkpoint_identity(output_dir, args):
    """Refuse to overwrite checkpoint metadata from a different ablation arm."""
    metadata_path = os.path.join(output_dir, "training-metadata.json")
    if not os.path.isfile(metadata_path):
        return
    try:
        with open(metadata_path) as f:
            existing = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"cannot safely inspect existing checkpoint metadata {metadata_path}: {exc}"
        ) from exc

    # Missing ``qh_aux`` denotes a checkpoint without the auxiliary Q_H view.
    existing_qh_aux = bool(existing.get("qh_aux", False))
    # Missing ``qy_loss`` on a Q_H-aux checkpoint denotes token CE. This prevents a
    # forced-choice run from overwriting a scientifically different checkpoint.
    existing_qy_loss = (
        existing.get("qy_loss", QY_LOSS_TOKEN_CE) if existing_qh_aux else None
    )
    # Grounded identity keys are null for non-grounded runs. Artifact/training-set
    # digests remain audit metadata rather than checkpoint-directory identity because a
    # valid teacher regeneration changes those digests.
    existing_grounded = existing.get("supervision") == SUPERVISION_GROUNDED
    existing_identity = (
        existing.get("dataset"),
        existing.get("mode"),
        existing.get("supervision", SUPERVISION_RATIONALE),
        existing_qh_aux,
        existing.get("qh_lambda"),
        existing_qy_loss,
        existing.get("lambda_lm") if existing_grounded else None,
        existing.get("lambda_fc") if existing_grounded else None,
        existing.get("grounded_prompt_version") if existing_grounded else None,
        existing.get("grounded_unresolved_policy") if existing_grounded else None,
    )
    requested_identity = (
        args.dataset,
        args.mode,
        supervision_of(args),
        qh_aux_of(args),
        qh_lambda_of(args) if qh_aux_of(args) else None,
        qy_loss_of(args) if qh_aux_of(args) else None,
        lambda_lm_of(args),
        lambda_fc_of(args),
        GROUNDED_PROMPT_VERSION if grounded_of(args) else None,
        grounded_unresolved_of(args),
    )
    if existing_identity != requested_identity:
        raise RuntimeError(
            f"refusing to overwrite {output_dir!r}: existing checkpoint identity "
            f"{existing_identity!r} != requested {requested_identity!r}. Choose a new "
            "--output-dir or deliberately move the existing checkpoint."
        )


def train(args):
    cfg = mode_config(args)
    supervision = supervision_of(args)
    qh_aux = qh_aux_of(args)
    qh_lambda = qh_lambda_of(args)
    qy_loss = qy_loss_of(args)
    grounded = grounded_of(args)
    lambda_lm = lambda_lm_of(args)
    lambda_fc = lambda_fc_of(args)
    grounded_unresolved = grounded_unresolved_of(args)
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    # The constant rationale is restricted to smoke and tokenizer validation.
    if getattr(args, "debug_constant_rationale", False) and not args.smoke:
        raise SystemExit("--debug-constant-rationale is only valid with --smoke or "
                         "--check-tokenizer; a real training run must use verified offline "
                         "rationale artifacts. Refusing to train on the constant scaffold.")
    if supervision == SUPERVISION_ANSWER_ONLY and getattr(
        args, "debug_constant_rationale", False
    ):
        raise SystemExit(
            "--debug-constant-rationale is incompatible with --supervision answer-only"
        )

    output_dir = effective_output_dir(args)
    guard_checkpoint_identity(output_dir, args)
    patch_multiprocess_resource_tracker_shutdown()
    preflight(world_size, use_deepspeed)

    import torch
    import transformers
    import accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

    class TokenizedRowsDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            # Project to the collator's keys only; continuation_ids/row_id are carried on
            # the tokenized records for the mask audit but must not reach the Trainer.
            return {"input_ids": r["input_ids"], "attention_mask": r["attention_mask"],
                    "labels": r["labels"]}

    class GroundedRowsDataset(torch.utils.data.Dataset):
        """One grounded example: the LM view plus its analysis-free direct readout."""

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            return {
                "input_ids": r["input_ids"],
                "attention_mask": r["attention_mask"],
                "labels": r["labels"],
                "qy_readout_input_ids": r["qy_readout_input_ids"],
                "qy_readout_attention_mask": r["qy_readout_attention_mask"],
                "qy_target_index": r["qy_target_index"],
                "grounded_lm_enabled": r.get("grounded_lm_enabled", True),
            }

    class QHPairedRowsDataset(torch.utils.data.Dataset):
        """One paired (Q_Y, Q_H) example per index; per-leg tensors are kept apart."""

        def __init__(self, qy_rows, qh_rows, qy_loss):
            if len(qy_rows) != len(qh_rows):
                raise RuntimeError(
                    f"paired dataset needs equal legs, got {len(qy_rows)}/{len(qh_rows)}"
                )
            for qy, qh in zip(qy_rows, qh_rows):
                if qh["qy_row_id"] != qy["row_id"]:
                    raise RuntimeError(
                        f"paired dataset misalignment: {qh['row_id']} claims "
                        f"{qh['qy_row_id']} but sits against {qy['row_id']}"
                    )
            self.qy_rows = qy_rows
            self.qh_rows = qh_rows
            self.qy_loss = qy_loss

        def __len__(self):
            return len(self.qy_rows)

        def __getitem__(self, idx):
            qy, qh = self.qy_rows[idx], self.qh_rows[idx]
            record = {
                "input_ids": qy["input_ids"],
                "attention_mask": qy["attention_mask"],
                # Keep labels even in forced-choice mode: Trainer uses their presence to
                # route labelled evaluation batches through compute_loss. The collator and
                # Trainer contract guarantees they are never fed to the FC model forward.
                "labels": qy["labels"],
                "qh_input_ids": qh["qh_input_ids"],
                "qh_attention_mask": qh["qh_attention_mask"],
                "qh_target_index": qh["qh_target_index"],
            }
            if self.qy_loss == QY_LOSS_FORCED_CHOICE:
                record.update(
                    qy_readout_input_ids=qy["qy_readout_input_ids"],
                    qy_readout_attention_mask=qy["qy_readout_attention_mask"],
                    qy_target_index=qy["qy_target_index"],
                )
            return record

    class GroundedQHPairedRowsDataset(QHPairedRowsDataset):
        """One grounded Q_Y row paired with its Q_H sibling.

        The Q_Y row already contains the grounded continuation/mask plus the direct
        readout. The extra flag is the only field the combined collator/trainer needs
        beyond the ordinary Q_H-paired dataset.
        """

        def __getitem__(self, idx):
            record = super().__getitem__(idx)
            record["grounded_lm_enabled"] = bool(
                self.qy_rows[idx].get("grounded_lm_enabled", True)
            )
            return record

    if args.effective_batch % world_size != 0:
        raise RuntimeError(f"--effective-batch {args.effective_batch} not divisible by "
                           f"world_size {world_size}.")
    grad_accum = args.effective_batch // world_size

    if args.smoke:
        if is_main:
            print(f"SMOKE MODE: 2 optimizer steps on 8 rows; saving to {output_dir} to "
                  "validate the full save path before the real run.")

    # TrainingArguments FIRST: its __post_init__ initializes distributed state and
    # exports the FSDP env (ACCELERATE_USE_FSDP / FSDP_CPU_RAM_EFFICIENT_LOADING)
    # that from_pretrained consults for rank0-only CPU loading.
    eval_strategy = "no" if args.train_all else "epoch"
    ta_kwargs = dict(
        output_dir=output_dir,
        report_to="none",
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        optim="adamw_torch_fused",
        logging_steps=1,
        eval_strategy=eval_strategy,
        save_strategy="no",  # ~88 steps total; one manual final save (FSDP dumps are huge)
        seed=3407,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if grounded:
        # ONLY on this arm. num_items_in_batch is never computed (model_accepts_loss_kwargs
        # is forced False), so cross-device token averaging would be dead code. The split
        # is KEPT for this arm, so Trainer evaluation actually runs: prediction_loss_only
        # makes the eval memory contract explicit instead of relying on compute_metrics
        # being None.
        ta_kwargs.update(average_tokens_across_devices=False, prediction_loss_only=True)
    if qh_aux:
        # ``num_items_in_batch`` is disabled for the paired objective, so cross-device
        # token averaging would add an unused collective. Both forced-choice legs are one
        # decision; token CE keeps its device-local continuation normalization.
        ta_kwargs.update(average_tokens_across_devices=False)
    if args.smoke:
        ta_kwargs.update(max_steps=2, eval_strategy="no")
    if not use_deepspeed:
        ta_kwargs.update(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": ["GptOssDecoderLayer"],
                "state_dict_type": "FULL_STATE_DICT",
                "use_orig_params": True,
                "cpu_ram_efficient_loading": True,
                "sync_module_states": True,
            },
        )
    training_args = TrainingArguments(**ta_kwargs)
    if qh_aux and getattr(training_args, "average_tokens_across_devices", None) is not False:
        raise RuntimeError(
            "TrainingArguments.average_tokens_across_devices did not stay False under "
            f"--qh-aux (got {getattr(training_args, 'average_tokens_across_devices', None)!r}); "
            "the paired Q_Y/Q_H objective requires explicit device-local normalization."
        )

    model_source = BF16_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data: rebuild, drift-guard against the audited artifacts, tokenize.
    # (Every rank needs the rows; only rank 0 narrates.)
    import contextlib
    import io
    with contextlib.redirect_stdout(sys.stdout if is_main else io.StringIO()):
        split, train_rows, eval_rows = build_all(write_artifacts=False, args=args)
        verify_artifacts_match(train_rows, cfg["train_jsonl"])
        verify_artifacts_match(eval_rows, cfg["eval_jsonl"])
        # Resolve the selected supervision target. Answer-only returns before touching
        # any rationale artifact; rationale mode verifies and attaches one per row.
        # messages/target always stay sourced from the audited rebuild.
        items = load_items_for_mode(args)
        grounded_fallback_rows = 0
        grounded_artifact_records = {"train": [], "eval": []}
        grounded_train_indices, grounded_eval_indices = [], []
        if grounded:
            grounded_fallback_rows = (
                resolve_grounded(train_rows, "train", items, args)
                + resolve_grounded(eval_rows, "eval", items, args)
            )
            for split_key, split_rows in (("train", train_rows), ("eval", eval_rows)):
                _, path, _ = _grounded_paths(split_key, args)
                with open(path) as f:
                    grounded_artifact_records[split_key] = [
                        json.loads(line) for line in f if line.strip()
                    ]
        else:
            resolve_rationales(train_rows, "train", items, args)
            resolve_rationales(eval_rows, "eval", items, args)
        qh_train_rows, qh_eval_rows, qh_audit = ([], [], None)
        # Artifact-aligned snapshots, kept BEFORE the --train-all merge so metadata can
        # hash what is actually on disk instead of the merged training set.
        qh_artifact_train_rows, qh_artifact_eval_rows = [], []
        if qh_aux:
            qh_train_rows, qh_eval_rows, qh_audit = build_qh_all(args, write_artifacts=False)
            verify_qh_artifacts_match(qh_train_rows, cfg["qh_train_jsonl"])
            verify_qh_artifacts_match(qh_eval_rows, cfg["qh_eval_jsonl"])
            qh_artifact_train_rows, qh_artifact_eval_rows = qh_train_rows, qh_eval_rows
        if args.train_all:
            print("TRAIN-ALL MODE: using audited train+eval rows for training; "
                  "Trainer evaluation dataset is disabled.")
            train_rows = train_rows + eval_rows
            eval_rows = []
            qh_train_rows = qh_train_rows + qh_eval_rows
            qh_eval_rows = []
    if args.smoke:
        train_rows = train_rows[:8]
    if grounded:
        # AFTER the --train-all merge and the --smoke truncation: metadata's
        # training_set_sha256 must cover exactly the items the Trainer consumes.
        grounded_train_indices = list(dict.fromkeys(r["dataset_index"] for r in train_rows))
        grounded_eval_indices = list(dict.fromkeys(r["dataset_index"] for r in eval_rows))

    train_tok = tokenize_rows(
        train_rows,
        tokenizer,
        args.max_len,
        supervision=supervision,
    )
    eval_tok = (
        []
        if args.train_all
        else tokenize_rows(
            eval_rows,
            tokenizer,
            args.max_len,
            supervision=supervision,
        )
    )
    if is_main:
        _check_unmasked_label_spans(
            train_tok,
            tokenizer,
            n=3,
            supervision=supervision,
        )
    train_ds = TokenizedRowsDataset(train_tok)
    eval_ds = None if args.train_all else TokenizedRowsDataset(eval_tok)

    grounded_letter_ids = grounded_prefill_ids = None
    if grounded:
        control_ids = harmony_control_ids(tokenizer)
        # Shared readout helpers: they depend only on the imported score_verifier
        # constants, not on --qh-aux, so the grounded direct view scores exactly the slot
        # the production evaluator reads.
        grounded_letter_ids = qh_letter_token_ids(tokenizer)
        grounded_prefill_ids = qh_prefill_token_ids(tokenizer, control_ids)
        assert_qy_answer_slot_letter_tokens(
            tokenizer, grounded_letter_ids, control_ids, prefill_ids=grounded_prefill_ids
        )
        attach_grounded_direct_readouts(
            train_rows, train_tok, grounded_prefill_ids, grounded_letter_ids,
            control_ids["<|return|>"], args.max_len,
        )
        if not args.train_all:
            attach_grounded_direct_readouts(
                eval_rows, eval_tok, grounded_prefill_ids, grounded_letter_ids,
                control_ids["<|return|>"], args.max_len,
            )
        # Run on every rank so a tokenization-contract failure cannot be rank-specific.
        assert_qy_chat_template_tokenization_parity(train_rows, train_tok, tokenizer, n=None)
        if not args.train_all:
            assert_qy_chat_template_tokenization_parity(eval_rows, eval_tok, tokenizer, n=None)
        if is_main:
            _check_qy_readout_spans(train_tok, tokenizer, n=2)
            print(f"Grounded arm ON: lambda_lm={lambda_lm}, lambda_fc={lambda_fc}, "
                  f"unresolved policy {grounded_unresolved!r}, "
                  f"{grounded_fallback_rows} fallback row(s), ' A'="
                  f"{grounded_letter_ids['A']} ' B'={grounded_letter_ids['B']}.")
        train_ds = GroundedRowsDataset(train_tok)
        eval_ds = None if args.train_all else GroundedRowsDataset(eval_tok)

    qy_continuation_token_count = None
    qh_letter_ids = qh_prefill_ids = None
    qh_train_tok = qh_eval_tok = []
    # The Q_H rows the Trainer actually consumes, AFTER the --train-all merge and the
    # --smoke truncation. Metadata's training-set hash is taken from exactly these.
    qh_train_selected, qh_eval_selected = [], []
    if qh_aux:
        control_ids = harmony_control_ids(tokenizer)
        qh_letter_ids = qh_letter_token_ids(tokenizer)
        qh_prefill_ids = qh_prefill_token_ids(tokenizer, control_ids)
        reference_c = assert_qy_answer_slot_letter_tokens(
            tokenizer, qh_letter_ids, control_ids, prefill_ids=qh_prefill_ids
        )
        if grounded:
            # Grounded rows intentionally contain an audited analysis continuation, so
            # their LM target is not the short answer-only C-token continuation. The
            # grounded direct Q_Y readout was already attached above; only the Q_H sibling
            # needs to be added here.
            qy_continuation_token_count = None
        else:
            qy_continuation_token_count = constant_continuation_tokens(train_tok + eval_tok)
            if qy_continuation_token_count != reference_c:
                raise RuntimeError(
                    f"answer-only continuation is {qy_continuation_token_count} tokens on real rows "
                    f"but {reference_c} on the reference render; the Q_Y readout tail is not pinned."
                )
        if qy_loss == QY_LOSS_FORCED_CHOICE:
            if not grounded:
                attach_qy_forced_choice_readouts(
                    train_rows, train_tok, qh_prefill_ids, qh_letter_ids,
                    control_ids["<|return|>"], args.max_len,
                )
                if not args.train_all:
                    attach_qy_forced_choice_readouts(
                        eval_rows, eval_tok, qh_prefill_ids, qh_letter_ids,
                        control_ids["<|return|>"], args.max_len,
                    )
            # Run on every rank so a tokenization-contract failure cannot be rank-specific.
            assert_qy_chat_template_tokenization_parity(
                train_rows, train_tok, tokenizer, n=None
            )
            if not args.train_all:
                assert_qy_chat_template_tokenization_parity(
                    eval_rows, eval_tok, tokenizer, n=None
                )
        qh_train_selected = select_qh_siblings(train_rows, qh_train_rows)
        qh_eval_selected = [] if args.train_all else select_qh_siblings(eval_rows, qh_eval_rows)
        qh_train_tok = tokenize_qh_rows(qh_train_selected, tokenizer,
                                        args.max_len, qh_prefill_ids, qh_letter_ids)
        qh_eval_tok = tokenize_qh_rows(qh_eval_selected, tokenizer,
                                       args.max_len, qh_prefill_ids, qh_letter_ids)
        if is_main:
            if qy_loss == QY_LOSS_FORCED_CHOICE:
                _check_qy_readout_spans(train_tok, tokenizer, n=2)
            _check_qh_readout_spans(qh_train_tok, tokenizer, n=2)
            weighting = (
                "lambda=1 is equal per-decision weighting"
                if qy_loss == QY_LOSS_FORCED_CHOICE
                else f"token-CE Q_Y over C={qy_continuation_token_count} tokens"
            )
            print(f"Q_H aux ON: qy_loss={qy_loss!r}, lambda={qh_lambda} ({weighting}), "
                  f"target semantic {cfg['qh_target_semantic']}, ' A'={qh_letter_ids['A']} "
                  f"' B'={qh_letter_ids['B']}.")
        train_ds = (
            GroundedQHPairedRowsDataset(train_tok, qh_train_tok, qy_loss)
            if grounded else QHPairedRowsDataset(train_tok, qh_train_tok, qy_loss)
        )
        eval_ds = (None if args.train_all
                   else (
                       GroundedQHPairedRowsDataset(eval_tok, qh_eval_tok, qy_loss)
                       if grounded else QHPairedRowsDataset(eval_tok, qh_eval_tok, qy_loss)
                   ))

    # Model: fp32 master weights under FSDP (bf16 compute via MixedPrecision from
    # bf16=True); bf16 under ZeRO-3 (DeepSpeed keeps fp32 masters internally).
    load_dtype = torch.bfloat16 if use_deepspeed else torch.float32
    if is_main:
        print(f"Loading {model_source} in {load_dtype} ...")
    model = _from_pretrained_compat(
        AutoModelForCausalLM, model_source, load_dtype,
        attn_implementation="eager",
        use_cache=False,
    )
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("model loaded WITH quantization_config — full FT requires the "
                           "dequantized base (run --prepare-bf16).")
    def logical_numel(param):
        # ZeRO-3 init can materialize parameters as local shards/placeholders;
        # DeepSpeed keeps the global size in ds_numel when param.numel() is 0.
        return int(getattr(param, "ds_numel", param.numel()))

    n_params = sum(logical_numel(p) for p in model.parameters())
    n_trainable = sum(logical_numel(p) for p in model.parameters() if p.requires_grad)
    if n_trainable != n_params:
        raise RuntimeError(f"only {n_trainable}/{n_params} params trainable — full FT "
                           "requires every parameter to receive gradients.")
    if is_main:
        print(f"Model loaded: {n_params / 1e9:.2f}B params, 100% trainable (full FT).")

    grounded_logits_kwarg = None
    if grounded:
        grounded_logits_kwarg = resolve_logits_to_keep_kwarg(model)
        if is_main:
            if grounded_logits_kwarg:
                print("Grounded direct-view forwards request only the last logits via "
                      f"{grounded_logits_kwarg!r}.")
            else:
                print("Grounded direct-view forwards fall back to FULL logits: this "
                      "transformers build exposes neither logits_to_keep nor "
                      "num_logits_to_keep on the model forward.")

    qh_logits_kwarg = None
    if qh_aux:
        qh_logits_kwarg = resolve_logits_to_keep_kwarg(model)
        if is_main:
            if qh_logits_kwarg:
                print(f"Forced-choice forwards request only the last logits via "
                      f"{qh_logits_kwarg!r}.")
            else:
                print("Forced-choice forwards fall back to FULL logits: this transformers build exposes "
                      "neither logits_to_keep nor num_logits_to_keep on the model forward.")
    if grounded and qh_aux:
        # Explicit three-view arm: grounded LM + direct grounded Q_Y forced choice + Q_H
        # positive control. It must not fall through to either two-view trainer.
        trainer = grounded_qh_aux_trainer_class(Trainer)(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=GroundedQHPairCollator(
                tokenizer.pad_token_id,
                qy_loss=QY_LOSS_FORCED_CHOICE,
                letter_ids=qh_letter_ids,
                prefill_ids=qh_prefill_ids,
                return_token_id=control_ids["<|return|>"],
            ),
            lambda_lm=lambda_lm,
            lambda_fc=lambda_fc,
            qh_lambda=qh_lambda,
            letter_ids=qh_letter_ids,
            logits_kwarg=grounded_logits_kwarg,
            smoke=args.smoke,
            readout_prefill_last_id=qh_prefill_ids[-1],
        )
        if trainer.model_accepts_loss_kwargs:
            raise RuntimeError(
                "GroundedQHAuxTrainer.model_accepts_loss_kwargs is True; num_items_in_batch "
                "would rescale one leg of the three-view loss."
            )
    elif qh_aux:
        trainer = qh_aux_trainer_class(Trainer)(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=QHPairCollator(
                tokenizer.pad_token_id,
                qy_loss=qy_loss,
                letter_ids=qh_letter_ids,
                prefill_ids=qh_prefill_ids,
                return_token_id=control_ids["<|return|>"],
            ),
            qh_lambda=qh_lambda,
            qh_letter_ids=qh_letter_ids,
            qh_logits_kwarg=qh_logits_kwarg,
            qh_smoke=args.smoke,
            qy_loss=qy_loss,
            readout_prefill_last_id=qh_prefill_ids[-1],
        )
        if trainer.model_accepts_loss_kwargs:
            raise RuntimeError(
                "QHAuxTrainer.model_accepts_loss_kwargs is True; num_items_in_batch would "
                "rescale L_Y alone."
            )
    elif grounded:
        trainer = grounded_trainer_class(Trainer)(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=GroundedCollator(
                tokenizer.pad_token_id,
                letter_ids=grounded_letter_ids,
                prefill_ids=grounded_prefill_ids,
                return_token_id=control_ids["<|return|>"],
            ),
            lambda_lm=lambda_lm,
            lambda_fc=lambda_fc,
            grounded_letter_ids=grounded_letter_ids,
            grounded_logits_kwarg=grounded_logits_kwarg,
            grounded_smoke=args.smoke,
            readout_prefill_last_id=grounded_prefill_ids[-1],
        )
        if trainer.model_accepts_loss_kwargs:
            raise RuntimeError(
                "GroundedTrainer.model_accepts_loss_kwargs is True; num_items_in_batch "
                "would rescale the LM view alone."
            )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=PadCollator(tokenizer.pad_token_id),
        )
    trainer.train()
    if qh_aux:
        if grounded:
            expected_qy_forwards = 0 if lambda_fc == 0.0 else None
            if expected_qy_forwards is not None and trainer.qy_forwards == 0:
                raise RuntimeError(
                    "the grounded direct Q_Y forward never ran; the combined objective "
                    "was inert."
                )
            if expected_qy_forwards == 0 and trainer.qy_forwards != 0:
                raise RuntimeError(
                    f"--lambda-fc 0 must skip the grounded direct Q_Y forward exactly, "
                    f"but it ran {trainer.qy_forwards} time(s)."
                )
        elif trainer.qy_forwards == 0:
            raise RuntimeError("the Q_Y forward never ran; the paired objective was inert.")
        expected_forwards = 0 if qh_lambda == 0.0 else None
        if expected_forwards is not None and trainer.qh_forwards != expected_forwards:
            raise RuntimeError(
                f"--qh-lambda 0 must skip the Q_H forward exactly, but it ran "
                f"{trainer.qh_forwards} time(s)."
            )
        if qh_lambda != 0.0 and trainer.qh_forwards == 0:
            raise RuntimeError("the Q_H forward never ran; the auxiliary loss was inert.")
    if grounded:
        if lambda_fc == 0.0:
            if trainer.fc_forwards != 0:
                raise RuntimeError(
                    f"--lambda-fc 0 must skip the grounded direct forced-choice forward "
                    f"exactly, but it ran {trainer.fc_forwards} time(s)."
                )
        elif trainer.fc_forwards == 0:
            raise RuntimeError("the direct forced-choice forward never ran; the grounded "
                               "objective was inert.")
        if lambda_lm == 0.0 and trainer.lm_forwards != 0:
            raise RuntimeError(
                f"--lambda-lm 0 must skip the grounded LM forward exactly, but it ran "
                f"{trainer.lm_forwards} time(s)."
            )
        if lambda_lm != 0.0 and trainer.lm_forwards == 0:
            raise RuntimeError("the grounded LM forward never ran; View 1 was inert.")

    # Final save: all ranks join the FULL_STATE_DICT gather; rank0 casts fp32 ->
    # bf16 (~42 GB instead of 84) and writes the checkpoint.
    wrapped = trainer.model_wrapped if trainer.model_wrapped is not None else trainer.model
    state_dict = trainer.accelerator.get_state_dict(wrapped)
    if trainer.is_world_process_zero():
        bf16_state = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        unwrapped = trainer.accelerator.unwrap_model(wrapped)
        unwrapped.save_pretrained(output_dir, state_dict=bf16_state, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "config.json")) as f:
            saved = json.load(f)
        assert "quantization_config" not in saved, "saved checkpoint carries quantization_config"

        if supervision == SUPERVISION_ANSWER_ONLY:
            forced_choice_qy = qh_aux and qy_loss == QY_LOSS_FORCED_CHOICE
            rationale_supervision = {
                "mode": "none-answer-only",
                "analysis_channel_supervised": False,
                "rationale_artifacts_used": False,
                "note": (
                    "No rationale was loaded, attached, or teacher-forced; Q_Y is supervised "
                    + ("only as a restricted A/B forced choice at the final-channel answer "
                       "slot."
                       if forced_choice_qy else
                       "by the direct final-channel Answer: A/B continuation.")
                ),
            }
            if forced_choice_qy:
                loss_mask = (
                    "Q_Y forced-choice branch: no token labels are passed to model(); input "
                    "ends at FINAL_CHANNEL_PREFILL and restricted NLL scores only ' A'/' B'. "
                    "The audited answer-only labels remain in the batch solely for Trainer "
                    "evaluation routing and fail-closed target/readout checks; no analysis "
                    "channel, rationale, format scaffold, or return token contributes loss"
                )
            else:
                loss_mask = (
                    "prompt masked; supervised continuation = direct harmony final-channel "
                    "format scaffold + 'Answer: X' + return token; no analysis channel or rationale"
                )
        elif supervision == SUPERVISION_GROUNDED:
            rationale_supervision = {
                "mode": "none-grounded",
                "analysis_channel_supervised": True,
                "rationale_artifacts_used": False,
                "note": (
                    "No rationale was loaded, attached, or teacher-forced. The analysis "
                    "channel carries a per-item, mechanically audited, orientation-"
                    "invariant grounded adjudication target (see the 'grounded' block)."
                    + (" --qh-aux additionally trains the explicit Q_H positive-control "
                       "readout; this is a three-view checkpoint."
                       if qh_aux else "")
                ),
            }
            loss_mask = (
                "prompt masked; View 1 supervised continuation = per-item audited grounded "
                "analysis + explicit final-channel transition + 'Answer: X' + return token "
                "(stock token-mean CE); PLUS View 2, a restricted A/B forced-choice NLL at "
                "the direct final-channel answer slot whose input is the SAME chat prompt "
                "followed by FINAL_CHANNEL_PREFILL with NO analysis in context. Fallback "
                "rows (unresolved items under --grounded-unresolved answer-only) carry the "
                "answer-only continuation and contribute View 2 only"
            )
        elif getattr(args, "debug_constant_rationale", False):
            rationale_supervision = {"mode": "debug-constant", "constant": ANALYSIS_TARGET}
            loss_mask = (
                "prompt masked; supervised continuation = debug constant analysis-channel "
                "scaffold + explicit final-channel transition + 'Answer: X' + return token"
            )
        else:
            rationale_supervision = {
                "mode": "offline-rationale",
                "train_artifact": cfg["rationale_train_jsonl"],
                "eval_artifact": cfg["rationale_eval_jsonl"],
                "train_manifest": cfg["rationale_manifest_template"].format(split="train"),
                "eval_manifest": cfg["rationale_manifest_template"].format(split="eval"),
                "train_all_uses_eval_artifact_as_training_data": args.train_all,
            }
            try:
                with open(cfg["rationale_manifest_template"].format(split="train")) as f:
                    man = json.load(f)
                with open(cfg["rationale_manifest_template"].format(split="eval")) as f:
                    man_eval = json.load(f)
                rationale_supervision.update(
                    teacher_model=man.get("teacher_model"),
                    teacher_backend=man.get("teacher_backend"),
                    rationale_prompt_version=man.get("rationale_prompt_version"),
                    rationale_min_gold_unique_trigram_hits=man.get(
                        "rationale_min_gold_unique_trigram_hits"
                    ),
                    train_combined_sha256=man.get("combined_sha256"),
                    eval_combined_sha256=man_eval.get("combined_sha256"),
                )
            except Exception:  # noqa: BLE001 (provenance is best-effort metadata)
                pass
            loss_mask = (
                "prompt masked; supervised continuation = per-example analysis-channel "
                "rationale + explicit final-channel transition + 'Answer: X' + return token"
            )

        grounded_metadata = {"enabled": False}
        if grounded:
            train_manifest_path = cfg["grounded_manifest_template"].format(split="train")
            eval_manifest_path = cfg["grounded_manifest_template"].format(split="eval")
            manifests = {}
            for key, path in (("train", train_manifest_path), ("eval", eval_manifest_path)):
                try:
                    with open(path) as f:
                        manifests[key] = json.load(f)
                except Exception:  # noqa: BLE001 (provenance is best-effort metadata)
                    manifests[key] = {}
            resolved_records = [
                r for records in grounded_artifact_records.values() for r in records
                if r.get("status") == "resolved"
            ]
            gold_records = [r for r in resolved_records if r.get("teacher_saw_gold")]
            unresolved_ids = [
                r["pair_id"] for records in grounded_artifact_records.values()
                for r in records if r.get("status") != "resolved"
            ]
            grounded_metadata = {
                "enabled": True,
                "lambda_lm": lambda_lm,
                "lambda_fc": lambda_fc,
                "lambda_note": GROUNDED_LAMBDA_NOTE,
                "loss_formula": GROUNDED_LOSS_FORMULA,
                "grounded_prompt_version": GROUNDED_PROMPT_VERSION,
                "teacher_output_schema": GROUNDED_TEACHER_OUTPUT_SCHEMA,
                "analysis_schema": GROUNDED_ANALYSIS_SCHEMA,
                "audit_thresholds": grounded_audit_thresholds(),
                "teacher_model": manifests["train"].get("teacher_model"),
                "teacher_backend": manifests["train"].get("teacher_backend"),
                "train_manifest_combined_sha256": manifests["train"].get("combined_sha256"),
                "eval_manifest_combined_sha256": manifests["eval"].get("combined_sha256"),
                "blind_count": len(resolved_records) - len(gold_records),
                "gold_count": len(gold_records),
                "gold_tier_fraction": (
                    len(gold_records) / len(resolved_records) if resolved_records else 0.0
                ),
                "unresolved_policy": grounded_unresolved,
                "unresolved_count": len(unresolved_ids),
                "unresolved_pair_ids": unresolved_ids,
                "grounded_fallback_rows": grounded_fallback_rows,
                "final_channel_prefill": FINAL_CHANNEL_PREFILL,
                "final_channel_prefill_token_ids": grounded_prefill_ids,
                "label_completions": dict(LABEL_COMPLETIONS),
                "label_completion_token_ids": grounded_letter_ids,
                "logits_to_keep_kwarg": grounded_logits_kwarg,
                "lm_logits_positions_returned": trainer.lm_logits_shape_seen,
                "fc_logits_positions_returned": trainer.fc_logits_shape_seen,
                "lm_forward_calls": trainer.lm_forwards,
                "fc_forward_calls": trainer.fc_forwards,
                "forward_count_scope": "training microbatches only; evaluation excluded",
                "qy_readout_source": (
                    "the audited Q_Y chat prompt followed by FINAL_CHANNEL_PREFILL; the "
                    "grounded analysis is NEVER in context for this view"
                ),
                "qh_absent": not qh_aux,
                "h_hygiene": (
                    GROUNDED_H_HYGIENE_NOTE
                    if not qh_aux else
                    GROUNDED_H_HYGIENE_NOTE + " The combined arm additionally carries an "
                    "explicit Q_H positive-control input in its separate Q_H loss leg; "
                    "that leg is not part of grounded View 1 or View 2."
                ),
                "average_tokens_across_devices": False,
                "model_accepts_loss_kwargs": False,
                "prediction_loss_only": True,
                "microbatch": (
                    "one grounded example; grounded LM view (skipped for fallback rows and "
                    "at lambda_lm=0), direct Q_Y forced-choice view, and explicit Q_H "
                    "forced-choice positive-control view, one backward on the weighted sum"
                    if qh_aux else
                    "one grounded example; the LM view forward (skipped for fallback rows "
                    "and at lambda_lm=0) then the direct forced-choice forward, one "
                    "backward on the weighted sum"
                ),
                "split_seed": SPLIT_SEED,
                **grounded_artifact_provenance_metadata(
                    cfg, args, grounded_artifact_records["train"],
                    grounded_artifact_records["eval"], grounded_train_indices,
                    grounded_eval_indices, args.train_all,
                ),
                "semantics_note": (
                    GROUNDED_QH_AUX_SEMANTICS_NOTE
                    if qh_aux else GROUNDED_SEMANTICS_NOTE
                ),
            }

        qhaux_metadata = {"enabled": False}
        if qh_aux:
            loss_mask += (
                "; PLUS a Q_H forced-choice term with lambda=" + f"{qh_lambda} "
                "(see qhaux.loss_formula)"
            )
            qhaux_metadata = {
                "enabled": True,
                "lambda": qh_lambda,
                "lambda_note": qh_aux_lambda_note(qy_loss),
                "qy_loss": qy_loss,
                "qy_continuation_token_count": qy_continuation_token_count,
                "qy_continuation_token_count_role": (
                    "structural/tokenizer audit only; it is not a loss denominator"
                    if qy_loss == QY_LOSS_FORCED_CHOICE else
                    "token-CE continuation loss denominator"
                ),
                "target_semantic": cfg["qh_target_semantic"],
                "target_semantic_policy":
                    "MODE LOCK, no CLI override in v1: honest -> H_true, adversarial -> H_false",
                "loss_formula": (
                    GROUNDED_QH_AUX_LOSS_FORMULA
                    if grounded else qh_aux_loss_formula(qy_loss)
                ),
                "readout_prompt_version": QH_READOUT_PROMPT_VERSION,
                "readout_template_sha256": QH_TEMPLATE_SHA256,
                "readout_source": "adversarial_transcript.score_verifier (imported read-only)",
                "final_channel_prefill": FINAL_CHANNEL_PREFILL,
                "final_channel_prefill_token_ids": qh_prefill_ids,
                "label_completions": dict(LABEL_COMPLETIONS),
                "label_completion_token_ids": qh_letter_ids,
                "qy_answer_slot_letter_token_ids": qh_letter_ids,
                "qy_readout_source": (
                    "the grounded View 2 readout: audited Q_Y chat prompt + transcript "
                    "followed by FINAL_CHANNEL_PREFILL with no grounded analysis in context"
                    if grounded else
                    "exact audited tokenized Q_Y input_ids with only the final target-letter "
                    "and canonical return-token ids removed; equivalently, the unchanged "
                    "Q_Y chat prompt + transcript followed by FINAL_CHANNEL_PREFILL"
                    if qy_loss == QY_LOSS_FORCED_CHOICE else
                    "model(labels=...).loss over the audited answer-only continuation"
                ),
                "qy_readout_prompt_template": (
                    "the grounded View 2 audited Q_Y chat prompt (same direct readout as the "
                    "grounded arm)"
                    if grounded else
                    "judge_common.build_judge_user_content_mapped (the audited training "
                    "prompt) — deliberately NOT score_verifier.QY_READOUT_TEMPLATE, which "
                    "the production Q_Y evaluator uses"
                    if qy_loss == QY_LOSS_FORCED_CHOICE else None
                ),
                "qy_labels_usage": (
                    "retained in batch for Trainer labelled-evaluation routing and collator "
                    "contract checks; used by the grounded LM View 1, but never passed to "
                    "the grounded direct Q_Y forced-choice forward"
                    if grounded else
                    "retained in batch for Trainer labelled-evaluation routing and collator "
                    "contract checks; never passed to model in the forced-choice branch"
                    if qy_loss == QY_LOSS_FORCED_CHOICE else
                    "passed to the model for token CE"
                ),
                "logits_to_keep_kwarg": qh_logits_kwarg,
                "qy_logits_positions_returned": trainer.qy_logits_shape_seen,
                "qh_logits_positions_returned": trainer.qh_logits_shape_seen,
                "qy_forward_calls": trainer.qy_forwards,
                "qh_forward_calls": trainer.qh_forwards,
                "forward_count_scope": "training microbatches only; evaluation excluded",
                "option_seed": QH_OPTION_SEED,
                "item_id_convention":
                    "adversarial_transcript.common.item_id_for(source_dataset_index or "
                    "dataset_index) — the canonical merged-dataset position",
                "schedule_version": QH_AUX_SCHEDULE_VERSION,
                "schedule_formula": (
                    'b = sha256(f"{SPLIT_SEED}|{pair_id}|' + QH_AUX_SCHEDULE_VERSION +
                    '").digest()[0] % 2; b=0 -> target letters (A, B) across the two pair '
                    "slots (ORIENTATIONS order), b=1 -> (B, A). The schedule is "
                    "MODE-INDEPENDENT; the mode's target semantic is placed at the scheduled "
                    "letter and the other H option opposite."
                ),
                "split_seed": SPLIT_SEED,
                # Artifact hashes come from the PRE-MERGE snapshots (so each matches its
                # file on disk); the training-set hash covers what the Trainer consumed.
                **qh_artifact_provenance_metadata(
                    cfg, qh_artifact_train_rows, qh_artifact_eval_rows,
                    qh_train_selected, qh_eval_selected, args.train_all,
                ),
                "alignment_audit": qh_audit,
                "average_tokens_across_devices": False,
                "model_accepts_loss_kwargs": False,
                "microbatch": (
                    "one grounded Q_Y row paired with one Q_H sibling; grounded LM forward, "
                    "direct grounded Q_Y forced-choice forward, then Q_H forced-choice "
                    "forward, one backward on the weighted sum"
                    if grounded else
                    "one paired (Q_Y, Q_H) example; two forced-choice forwards using the "
                    "same A/B NLL (Y then H), one backward on the weighted sum"
                    if qy_loss == QY_LOSS_FORCED_CHOICE else
                    "one paired (Q_Y, Q_H) example; token-CE Y forward then "
                    "forced-choice H forward, one backward on the weighted sum"
                ),
                "semantics_note": (
                    GROUNDED_QH_AUX_SEMANTICS_NOTE
                    if grounded else QH_AUX_SEMANTICS_NOTE
                ),
            }

        metadata = {
            "model_name": MODEL_NAME,
            "dataset": args.dataset,
            "mode": args.mode,
            "supervision": supervision,
            # Read by ``guard_checkpoint_identity``.
            "qh_aux": qh_aux,
            "qh_lambda": qh_lambda if qh_aux else None,
            "qy_loss": qy_loss if qh_aux else None,
            # Grounded identity keys, also read by guard_checkpoint_identity.
            "lambda_lm": lambda_lm,
            "lambda_fc": lambda_fc,
            "grounded_prompt_version": GROUNDED_PROMPT_VERSION if grounded else None,
            "grounded_unresolved_policy": grounded_unresolved,
            "supervised_continuation_template":
                (None if qh_aux and qy_loss == QY_LOSS_FORCED_CHOICE and not grounded
                 else supervised_continuation_template(supervision)),
            "audited_answer_only_continuation_template":
                (supervised_continuation_template(supervision)
                 if qh_aux and qy_loss == QY_LOSS_FORCED_CHOICE and not grounded else None),
            "forced_choice_readout_input_suffix":
                (FINAL_CHANNEL_PREFILL
                 if qh_aux and qy_loss == QY_LOSS_FORCED_CHOICE else None),
            "dataset_path": cfg["dataset_path"],
            "stories_path": cfg["stories_path"],
            "transcript_field": cfg["transcript_field"],
            "base_dir": model_source,
            "output_dir": output_dir,
            "smoke": args.smoke,
            "train_all": args.train_all,
            "split_seed": SPLIT_SEED,
            "train_items": len({r["dataset_index"] for r in train_rows}),
            "eval_items": 0 if args.train_all else split["eval_count"],
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "trainer_eval_dataset": not args.train_all,
            "answer_order_policy": "paired orientations: every item rendered as "
                                   "A=Y_true (target A) AND A=Y_false (target B)",
            "dtype_policy": ("bf16 load; DeepSpeed ZeRO-3 keeps fp32 masters" if use_deepspeed
                             else "fp32 master weights + bf16 MixedPrecision via FSDP"),
            "distributed": {"path": "deepspeed-zero3" if use_deepspeed else "fsdp",
                            "world_size": world_size},
            "hyperparameters": {
                "learning_rate": args.lr,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch": args.effective_batch,
                "warmup_steps": args.warmup,
                "weight_decay": args.weight_decay,
                "lr_scheduler_type": "cosine",
                "optim": "adamw_torch_fused",
                "max_seq_len": args.max_len,
                "seed": 3407,
                "loss_mask": loss_mask,
            },
            "rationale_supervision": rationale_supervision,
            "grounded": grounded_metadata,
            "qhaux": qhaux_metadata,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
            },
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_history": trainer.state.log_history,
        }
        write_json_atomic(os.path.join(output_dir, "training-metadata.json"), metadata)
        print(f"Saved full bf16 checkpoint + tokenizer + training-metadata.json to {output_dir}")
    trainer.accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _ExplicitRationaleValue(argparse.Action):
    """Store an option value and remember that this rationale-only flag was supplied."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        seen = set(getattr(namespace, "explicit_rationale_flags", set()))
        seen.add(option_string)
        setattr(namespace, "explicit_rationale_flags", seen)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, metavar="NAME",
        help="dataset directory/name under dataset/<NAME>/; expects "
             "<NAME>.json, <NAME>-with-honest-transcripts.json, and "
             "<NAME>-title-story.json (default %(default)s)"
    )
    parser.add_argument("--mode", choices=tuple(MODE_CONFIG), default="honest",
                        help="which Q_Y transcript set to prepare/train on: honest keeps "
                             "the existing artifacts; adversarial uses non-null "
                             "Q_Y.adversarial_transcript rows from <NAME>.json "
                             "(default %(default)s)")
    parser.add_argument(
        "--supervision",
        choices=SUPERVISION_MODES,
        default=SUPERVISION_RATIONALE,
        help=(
            "training target: rationale uses an analysis+rationale+final continuation; "
            "answer-only goes directly to the final channel and supervises "
            "only its format scaffold plus Answer: A/B (no rationale artifacts are read "
            "or written); grounded supervises a per-item, mechanically audited, "
            "orientation-invariant evidence/check/winner analysis PLUS a restricted A/B "
            "forced-choice loss at the direct answer slot (no Q_H term unless --qh-aux "
            "is supplied). Default: %(default)s"
        ),
    )
    parser.add_argument("--check-data", action="store_true",
                        help="build split + rows, run torch-free leak/sanity checks, "
                             "write split JSON + audit JSONLs, print samples, exit")
    parser.add_argument("--check-tokenizer", action="store_true",
                        help="tokenizer-only (cluster): tokenize all rows, verify mask "
                             "spans + token lengths, exit")
    parser.add_argument("--prepare-bf16", action="store_true",
                        help="one-time (cluster): dequantize MXFP4 -> bf16 and save the "
                             f"quantizer-free base to {BF16_DIR}")
    parser.add_argument("--generate-rationales", action="store_true",
                        help="(cluster/GPU) generate + validate the per-example rationale "
                             "targets for --split and write the rationale jsonl + manifest")
    parser.add_argument("--check-rationales", action="store_true",
                        help="(torch-free) verify both rationale artifacts (alignment, "
                             "hashes, manifest, guardrails) against the rebuilt base rows")
    parser.add_argument("--generate-grounded", action="store_true",
                        help="(cluster/GPU) generate + audit the per-item grounded "
                             "adjudication artifacts for --split and write the grounded "
                             "jsonl + manifest (requires --supervision grounded)")
    parser.add_argument(
        "--generate-grounded-workers", type=int, default=None, metavar="N",
        help=f"local Transformers worker count for --generate-grounded (default "
             f"{GROUNDED_WORKERS_DEFAULT}); use when no vLLM server is available. "
             "N>1 spawns one isolated process "
             "per GPU (each pins itself with CUDA_VISIBLE_DEVICES) and runs N per-item tier "
             "ladders concurrently; the artifact stays in canonical item order with a "
             "single writer, and seeds stay per-(row_id, attempt, epoch). With "
             "--teacher-backend api use --generate-grounded-concurrency instead. "
             "Requires --generate-grounded")
    parser.add_argument(
        "--generate-grounded-concurrency", type=int, default=None, metavar="N",
        help="bounded in-flight HTTP requests for --generate-grounded on --teacher-backend "
             f"api (default {GROUNDED_CONCURRENCY_DEFAULT} on that backend, 1 on every "
             "other backend, where it is rejected above 1). One shared server, N per-item "
             "tier ladders in flight; the artifact stays in canonical item order with a "
             "single writer, and seeds stay per-(row_id, attempt, epoch) and ARE sent to "
             "the server. Note the resume window widens: up to N in-flight items are "
             "unpersisted if the run dies (vs 1 per worker on the process path). "
             "Requires --generate-grounded")
    parser.add_argument("--check-grounded", action="store_true",
                        help="(torch-free) verify both grounded artifacts (order, key "
                             "sets, hashes, manifest, every audit) and print the audit "
                             "report (requires --supervision grounded)")
    parser.add_argument("--split", choices=("train", "eval"),
                        action=_ExplicitRationaleValue,
                        help="which split to generate targets for (required with "
                             "--generate-rationales / --generate-grounded)")
    parser.add_argument("--debug-constant-rationale", action="store_true",
                        help="use the constant ANALYSIS_TARGET instead of offline rationales; "
                             "ONLY valid with --smoke / --check-tokenizer (never a real run)")
    parser.add_argument("--teacher-backend", choices=("transformers", "api"),
                        action=_ExplicitRationaleValue,
                        default=TEACHER_BACKEND_DEFAULT,
                        help="rationale teacher backend (default %(default)s)")
    parser.add_argument("--teacher-model", action=_ExplicitRationaleValue,
                        default=TEACHER_MODEL_DEFAULT,
                        help="teacher model path/name (default %(default)s)")
    parser.add_argument("--teacher-base-url", action=_ExplicitRationaleValue,
                        default=TEACHER_BASE_URL_DEFAULT,
                        help="OpenAI-compatible base_url for --teacher-backend api")
    parser.add_argument("--teacher-max-new-tokens", type=int,
                        action=_ExplicitRationaleValue,
                        default=TEACHER_MAX_NEW_TOKENS_DEFAULT,
                        help="teacher generation cap (default %(default)s; must be large enough "
                             "to fit the gpt-oss analysis CoT before the final channel)")
    parser.add_argument("--teacher-api-key-env", action=_ExplicitRationaleValue,
                        default=TEACHER_API_KEY_ENV_DEFAULT,
                        help="env var holding the api key (value defaults to EMPTY for vLLM)")
    parser.add_argument(
        "--qh-aux", action="store_true",
        help="EXPLICIT POSITIVE CONTROL (default OFF): add a directly supervised Q_H "
             "forced-choice term, built from the production score_verifier readout. By "
             "default answer-only Q_Y is changed to the symmetric single-letter "
             "forced-choice loss; with --supervision grounded it adds Q_H to the existing "
             "grounded LM + direct Q_Y views. Requires --supervision answer-only or grounded. "
             "Writes/reads "
             "separate *-qh-{train,eval}.jsonl artifacts and a '-qhaux' checkpoint dir. "
             "This arm is NOT evidence of transcript-only emergent steering")
    parser.add_argument(
        "--qy-loss", choices=QY_LOSS_MODES, default=None,
        help="Q_Y objective inside --qh-aux: forced-choice (default) applies the exact same "
             "single-letter restricted A/B NLL as Q_H; token-ce selects the C-token "
             "answer-only Q_Y objective. Has no meaning without --qh-aux")
    parser.add_argument(
        "--qh-lambda", type=float, default=None, metavar="LAMBDA",
        help=f"weight of the Q_H term (default {QH_AUX_DEFAULT_LAMBDA}; requires --qh-aux). "
             "With --supervision grounded, this is the third-view Q_H weight and the default "
             "2 reflects the two Y_true-supporting grounded views. In answer-only forced-choice "
             "mode, lambda=1 is the equal-per-decision comparison value (not necessarily equal "
             "parameter-gradient norms). With token CE, the Q_Y letter is averaged over C "
             "tokens. 0 is a smoke-only plumbing value that skips the Q_H forward entirely")
    parser.add_argument(
        "--lambda-lm", type=float, default=None, metavar="LAMBDA",
        help=f"weight of the grounded LM view (default {GROUNDED_LAMBDA_LM_DEFAULT}; "
             "requires --supervision grounded). 0 is a --smoke-only plumbing value that "
             "skips the LM forward entirely")
    parser.add_argument(
        "--lambda-fc", type=float, default=None, metavar="LAMBDA",
        help=f"weight of the direct forced-choice view (default "
             f"{GROUNDED_LAMBDA_FC_DEFAULT}; requires --supervision grounded). It is the "
             "only term training the UNCONDITIONED decision at the deployed readout slot; "
             "0 is a --smoke-only plumbing value")
    parser.add_argument(
        "--grounded-unresolved", choices=GROUNDED_UNRESOLVED_POLICIES, default=None,
        help=f"what to do with items the grounded teacher never resolved (default "
             f"{GROUNDED_UNRESOLVED_DEFAULT}; requires --supervision grounded). fail "
             "refuses to train on a partially-resolved pool AFTER the artifact and its "
             "failure reasons are on disk; answer-only trains those items on the direct "
             "forced-choice view only. Neither policy drops an item")
    parser.add_argument("--smoke", action="store_true",
                        help="2 optimizer steps on 8 rows + the full save path, to a "
                             "-smoke dir (validate FSDP save before the real run)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="full-FT learning rate (gpt-oss-recipes full SFT uses 2e-5 "
                             "on far more data; 5e-6 is the conservative fallback)")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train-all", action="store_true",
                        help="for real training, merge the audited train+eval splits into "
                             "the training set and disable Trainer evaluation; rationale "
                             "artifacts are verified only in rationale supervision")
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--output-dir", default=None,
                        help="checkpoint output dir (default is selected by "
                             "--dataset/--mode/--supervision; "
                             f"{DEFAULT_OUTPUT_DIR.format(dataset=DEFAULT_DATASET)!r} "
                             "for default rationale/honest runs and "
                             f"{DEFAULT_ANSWER_ONLY_OUTPUT_DIR.format(dataset=DEFAULT_DATASET)!r} "
                             "for answer-only/honest runs)")
    args = parser.parse_args()

    # Q_H auxiliary guards run FIRST: they resolve args.qh_lambda, which mode_config
    # and every downstream path read. Grounded+Q_H is intentionally a separate combined
    # three-view arm; its Q_Y leg is always the grounded direct forced-choice view.
    if args.qh_aux and args.supervision not in (
        SUPERVISION_ANSWER_ONLY, SUPERVISION_GROUNDED
    ):
        sys.exit(
            "--qh-aux requires --supervision answer-only or grounded. The Q_H term is "
            "only supported by those explicitly audited Q_Y objectives."
        )
    if args.qy_loss is not None and not args.qh_aux:
        sys.exit("--qy-loss has no effect without --qh-aux; pass both or neither.")
    if args.qh_lambda is not None and not args.qh_aux:
        sys.exit("--qh-lambda has no effect without --qh-aux; pass both or neither.")
    if args.qh_aux:
        if args.qy_loss is None:
            args.qy_loss = QY_LOSS_DEFAULT
        if (args.supervision == SUPERVISION_GROUNDED
                and args.qy_loss != QY_LOSS_FORCED_CHOICE):
            sys.exit(
                "--supervision grounded --qh-aux always uses the grounded direct "
                "Q_Y forced-choice view; --qy-loss token-ce is not compatible with "
                "the three-view objective."
            )
        if args.qh_lambda is None:
            args.qh_lambda = QH_AUX_DEFAULT_LAMBDA
        if not math.isfinite(args.qh_lambda) or args.qh_lambda < 0:
            sys.exit(f"--qh-lambda must be a finite, non-negative float, got {args.qh_lambda!r}.")
        if args.qh_lambda == 0 and not args.smoke:
            sys.exit(
                "--qh-lambda 0 skips the Q_H forward entirely; it is a --smoke plumbing check "
                "(it must never produce a real checkpoint that claims to be a Q_H arm)."
            )

    # Grounded guards, in the same before-any-I/O position as the Q_H ones. The three
    # grounded knobs default to None, so "supplied" is distinguishable from "default"
    # exactly the way --qh-lambda / --qy-loss already are.
    grounded_only = {
        "--lambda-lm": args.lambda_lm,
        "--lambda-fc": args.lambda_fc,
        "--grounded-unresolved": args.grounded_unresolved,
    }
    supplied = [flag for flag, value in grounded_only.items() if value is not None]
    if supplied and args.supervision != SUPERVISION_GROUNDED:
        sys.exit(f"{', '.join(supplied)} only take effect with --supervision grounded; "
                 "omit them otherwise.")
    if (args.generate_grounded or args.check_grounded) and \
            args.supervision != SUPERVISION_GROUNDED:
        sys.exit("--generate-grounded / --check-grounded require --supervision grounded.")
    if args.generate_grounded_workers is not None:
        if not args.generate_grounded:
            sys.exit("--generate-grounded-workers only applies to --generate-grounded; "
                     "every other mode is single-process.")
        if args.generate_grounded_workers < 1:
            sys.exit("--generate-grounded-workers must be >= 1 (1 = the single-GPU, "
                     f"single-threaded path), got {args.generate_grounded_workers}.")
    if args.generate_grounded_concurrency is not None:
        if not args.generate_grounded:
            sys.exit("--generate-grounded-concurrency only applies to --generate-grounded; "
                     "no other mode issues teacher requests.")
        if args.generate_grounded_concurrency < 1:
            sys.exit("--generate-grounded-concurrency must be >= 1 (1 = serial requests), "
                     f"got {args.generate_grounded_concurrency}.")
        if args.generate_grounded_concurrency > 1 and args.teacher_backend != "api":
            sys.exit(
                "--generate-grounded-concurrency > 1 requires --teacher-backend api: the "
                "transformers teacher is one in-process replica whose seeded sampling uses "
                "the process-global torch RNG, so it must stay single-threaded. Use "
                "--generate-grounded-workers for the multi-GPU transformers path."
            )
    if (args.generate_grounded and args.teacher_backend == "api"
            and grounded_workers_of(args) > 1):
        sys.exit(
            "--generate-grounded-workers > 1 selects local multi-GPU Transformers workers and "
            "cannot be combined with --teacher-backend api: it would multiply to "
            f"workers x concurrency ({grounded_workers_of(args)} x "
            f"{grounded_concurrency_of(args)}) in-flight requests against one server. Use "
            "--generate-grounded-concurrency alone."
        )
    if args.supervision == SUPERVISION_GROUNDED:
        for flag, value in (("--lambda-lm", lambda_lm_of(args)),
                            ("--lambda-fc", lambda_fc_of(args))):
            if not math.isfinite(value) or value < 0:
                sys.exit(f"{flag} must be a finite, non-negative float, got {value!r}.")
            if value == 0 and not args.smoke:
                sys.exit(
                    f"{flag} 0 makes one of the two grounded views inert; it is a --smoke "
                    "plumbing check and must never produce a real grounded checkpoint."
                )
        incompatible = []
        if args.generate_rationales:
            incompatible.append("--generate-rationales")
        if args.check_rationales:
            incompatible.append("--check-rationales")
        if args.debug_constant_rationale:
            incompatible.append("--debug-constant-rationale")
        if args.qy_loss is not None and not args.qh_aux:
            incompatible.append("--qy-loss")
        if incompatible:
            sys.exit(
                "--supervision grounded is incompatible with option(s): "
                f"{', '.join(dict.fromkeys(incompatible))}. The grounded arm reads no "
                "rationale artifacts; --qh-aux is allowed only as the explicit three-view "
                "positive-control extension and uses forced-choice Q_Y."
            )

    cfg = mode_config(args)
    if args.output_dir is None:
        args.output_dir = cfg["output_dir"]

    modes = (args.check_data, args.check_tokenizer, args.prepare_bf16,
             args.generate_rationales, args.check_rationales,
             args.generate_grounded, args.check_grounded)
    if sum(modes) > 1:
        sys.exit("--check-data / --check-tokenizer / --prepare-bf16 / --generate-rationales "
                 "/ --check-rationales / --generate-grounded / --check-grounded are "
                 "mutually exclusive.")
    if args.supervision == SUPERVISION_ANSWER_ONLY:
        incompatible = []
        if args.generate_rationales:
            incompatible.append("--generate-rationales")
        if args.check_rationales:
            incompatible.append("--check-rationales")
        if args.debug_constant_rationale:
            incompatible.append("--debug-constant-rationale")
        if args.generate_grounded:
            incompatible.append("--generate-grounded")
        if args.check_grounded:
            incompatible.append("--check-grounded")
        incompatible.extend(sorted(getattr(args, "explicit_rationale_flags", set())))
        if incompatible:
            sys.exit(
                "--supervision answer-only is incompatible with rationale-only option(s): "
                f"{', '.join(dict.fromkeys(incompatible))}. Remove them; answer-only "
                "does not generate, load, check, or attach rationale targets."
            )
    if args.generate_rationales and not args.split:
        sys.exit("--generate-rationales requires --split {train,eval}.")
    if args.generate_grounded and not args.split:
        sys.exit("--generate-grounded requires --split {train,eval}.")
    if args.debug_constant_rationale and not (args.smoke or args.check_tokenizer):
        sys.exit("--debug-constant-rationale is only valid with --smoke or --check-tokenizer "
                 "(it must never produce a real checkpoint).")

    if args.check_data:
        _, train_rows, _ = build_all(write_artifacts=True, args=args)
        print_sample_prompts(train_rows)
        if args.qh_aux:
            qh_train_rows, _, _ = build_qh_all(args, write_artifacts=True)
            print_sample_qh_prompts(qh_train_rows)
        selection_flags = f" --dataset {args.dataset}"
        if args.mode != "honest":
            selection_flags += f" --mode {args.mode}"
        if args.qh_aux:
            selection_flags += f" --qh-aux --qy-loss {qy_loss_of(args)}"
        if args.supervision == SUPERVISION_GROUNDED:
            selection_flags += " --supervision grounded"
            print(
                f"\n--check-data complete for dataset={args.dataset!r}, "
                f"mode={args.mode!r}, supervision={args.supervision!r} "
                "(no training run). Base audit artifacts are shared with the other arms. "
                "Generate grounded targets next: "
                f"`--generate-grounded{selection_flags} --split train|eval` (cluster), "
                f"then `--check-grounded{selection_flags}` and "
                f"`--check-tokenizer{selection_flags}`."
            )
        elif args.supervision == SUPERVISION_ANSWER_ONLY:
            selection_flags += " --supervision answer-only"
            print(
                f"\n--check-data complete for dataset={args.dataset!r}, "
                f"mode={args.mode!r}, supervision={args.supervision!r} "
                "(no training run). Base audit artifacts are shared with the rationale "
                "arm; no rationale artifacts are needed. Run "
                f"`--check-tokenizer{selection_flags}` next."
            )
        else:
            print(f"\n--check-data complete for dataset={args.dataset!r}, "
                  f"mode={args.mode!r} (no training run). "
                  "Generate rationale targets next: "
                  f"`--generate-rationales{selection_flags} --split train|eval` (cluster), "
                  f"then `--check-rationales{selection_flags}` and "
                  f"`--check-tokenizer{selection_flags}`.")
        return
    if args.generate_rationales:
        generate_rationales(args)
        return
    if args.check_rationales:
        check_rationales(args)
        return
    if args.generate_grounded:
        generate_grounded(args)
        return
    if args.check_grounded:
        check_grounded(args)
        return
    if args.check_tokenizer:
        check_tokenizer(args)
        return
    if args.prepare_bf16:
        prepare_bf16()
        return

    train(args)


if __name__ == "__main__":
    main()
