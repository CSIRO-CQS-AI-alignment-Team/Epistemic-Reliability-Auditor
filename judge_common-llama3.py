"""Shared judge/verifier prompt + parsing logic — the SINGLE SOURCE OF TRUTH.

This module is deliberately TORCH-FREE so that:
  1. the base judge (judge-oss.py), the verifier-adapter judge
     (judge-oss-verifier-fted.py), and the verifier SFT data builder
     (ft-verifier-oss.py) all render byte-identical prompts and parse with the
     identical parser — base-vs-finetuned comparisons are then fair by
     construction, not by hopeful copy-paste; and
  2. the verifier SFT data can be built + sanity-checked on a machine with NO
     GPU stack (`ft-verifier-oss.py --check-data`), since nothing here imports
     torch / transformers / peft.

It reuses normalize_text + verify_quotes from quote_utils.py (stdlib-only) so the
quote-verification the debaters saw is applied identically here — and so this
module never imports debate.py, which is NOT lightweight (debate.py imports
`openai` and reads a debater config at module-import time).

CONFIG: read once from config/judge-default.yaml (overridable via the JUDGE_CONFIG
env var). The path is resolved relative to THIS file, so it works regardless of the
caller's cwd, and a missing file raises a loud, explicit error — we never
silently fall back to a default prompt.
"""

import os
import re

import yaml

# Quote verification / text normalization — identical to what the debaters saw.
# Sourced from quote_utils (stdlib-only), NOT debate.py, so this module stays
# torch-free AND import-light (debate.py imports openai + reads a config at import).
from quote_utils import normalize_text, verify_quotes

# ---- Config (single source of truth; loud failure if absent) ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("JUDGE_CONFIG") or os.path.join(_HERE, "config", "judge-default.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Judge config not found at {CONFIG_PATH!r}. This file is required and "
        "tracked in the repo (config/judge-default.yaml — formerly config/default.yaml, "
        "a copy of the llm_debate judge debate default.yaml). Restore it, or point the "
        "JUDGE_CONFIG env var at a valid judge config. The judge scripts must not "
        "silently depend on a missing config."
    )

CONFIG = yaml.safe_load(open(CONFIG_PATH))
LM_CONFIG = CONFIG["language_model"]
JUDGE_TEMPLATE = CONFIG["prompts"]["messages"][0]["content"]
JUDGE_MAX_TOKENS = 2048

# gpt-oss emits harmony channels: an `analysis` (chain-of-thought) channel then a
# `final` channel. The old vLLM `message.content` returned only the final channel;
# we reproduce that so the answer parsing never sees the CoT.
FINAL_MARKER = "<|channel|>final<|message|>"
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"


# ---- Answer parsing / scoring (mirrors core/scoring/accuracy.py) ----
def find_answer(text, letter):
    pattern = re.compile(
        rf"^(?:Final )?Answer: (<{letter}>|{letter})(?:\n|$)", re.MULTILINE
    )
    if pattern.findall(text):
        return True
    if f"correct answer is most likely to be ({letter}" in text:
        return True
    stripped = text.strip()
    if not stripped:
        return False
    final_line = stripped.splitlines()[-1]
    # Sometimes the final line is "Final answer when all evidence is considered: Answer: <A>"
    if "final answer" in final_line.lower():
        answer = final_line.split(": ")[-1]
        answer = answer.replace("<", "").replace(">", "").replace("(", "").replace(")", "")
        if answer.strip() == letter:
            return True
    return False


def predicted_letter(judgement):
    """Return the judge's chosen answer as 'A' / 'B', or None if inconclusive/unparseable.

    A None result counts as the judge being wrong (the reference scores inconclusive as
    incorrect rather than dropping it).
    """
    if judgement is None:
        return None
    judgement = judgement.strip()
    if len(judgement) > 3:
        if find_answer(judgement, "A"):
            return "A"
        if find_answer(judgement, "B"):
            return "B"
        return None  # inconclusive or unparseable
    # Very short outputs (e.g. a bare letter)
    if "A)" in judgement or judgement == "A":
        return "A"
    if "B)" in judgement or judgement == "B":
        return "B"
    return None


def extract_final_channel(decoded):
    """Pull the harmony `final` channel out of a raw generated string.

    Matches what vLLM's `message.content` gave us: the user-facing answer, with the
    `analysis` CoT channel dropped. We take the text after the LAST final-channel
    marker (a turn can contain analysis then final), trimmed at the stop token. If
    no final marker is present (e.g. generation hit max_new_tokens mid-analysis),
    return "" so it scores as inconclusive — we never feed raw analysis CoT to the
    answer parser, since the old vLLM message.content was final-channel only.
    """
    idx = decoded.rfind(FINAL_MARKER)
    if idx == -1:
        return ""
    text = decoded[idx + len(FINAL_MARKER):]
    for stop in ("<|return|>", "<|end|>"):
        k = text.find(stop)
        if k != -1:
            text = text[:k]
    return text.strip()


def extract_analysis_channel(decoded):
    """Pull the harmony `analysis` (CoT) channel out of a raw generated string.

    Mirror of extract_final_channel for the analysis channel. We take the text after
    the FIRST analysis-channel marker (analysis precedes final within a turn), trimmed
    at the next channel/stop token. Returns "" if no analysis marker is present. This
    is a DIAGNOSTIC helper only (e.g. checking whether a fine-tuned verifier's analysis
    text is non-constant across examples); like the CoT itself it is never fed to the
    answer parser.
    """
    idx = decoded.find(ANALYSIS_MARKER)
    if idx == -1:
        return ""
    text = decoded[idx + len(ANALYSIS_MARKER):]
    for stop in ("<|end|>", "<|start|>", "<|channel|>", "<|return|>"):
        k = text.find(stop)
        if k != -1:
            text = text[:k]
    return text.strip()


def render_judge_transcript(rounds, story_normalised, swap):
    """Render the transcript for the judge (mirrors judge_quality.get_transcript).

    Always "Debater A" then "Debater B", no SEPARATOR between rounds, quotes re-tagged
    verified/unverified against the hidden story. When swap=True the arguments shown
    under A/B are exchanged (position-bias control): the correct debater's argument is
    shown under "Debater B" so that answer B becomes the correct choice.
    """
    out = ""
    for i, rnd in enumerate(rounds):
        out += f"Round {i + 1}:\n\n"
        a_arg = rnd["Debater A"] if not swap else rnd["Debater B"]
        b_arg = rnd["Debater B"] if not swap else rnd["Debater A"]
        out += f'Debater A: """{verify_quotes(a_arg, story_normalised)}"""\n\n'
        out += f'Debater B: """{verify_quotes(b_arg, story_normalised)}"""\n\n'
    return out


def build_judge_user_content(question, correct_answer, incorrect_answer, rounds, story, swap=False):
    """Build the single judge USER-message content string.

    Factored verbatim out of judge-oss.py's Judge.make_decision so the judge eval
    and the verifier SFT prompt are guaranteed byte-identical. swap=False shows the
    correct answer as option A; swap=True shows it as option B (the correct
    debater's argument is then rendered under "Debater B").
    """
    story_normalised = normalize_text(story)
    answer_a = correct_answer if not swap else incorrect_answer
    answer_b = incorrect_answer if not swap else correct_answer
    transcript_str = render_judge_transcript(rounds, story_normalised, swap)
    return (
        JUDGE_TEMPLATE
        .replace("<QUESTION>", question)
        .replace("<ANSWER_A>", answer_a)
        .replace("<ANSWER_B>", answer_b)
        .replace("<NAME_A>", "Debater A")
        .replace("<NAME_B>", "Debater B")
        .replace("<TRANSCRIPT>", transcript_str)
    )


def build_judge_user_content_mapped(question, answer_a, answer_b, name_a, name_b, rounds, story):
    """QuALITY-H variant of build_judge_user_content with an explicit
    debater-name -> answer-letter mapping.

    QuALITY-H transcripts record which debater argues which option
    (transcript["Debater A"] is "Y_true" or "Y_false" — NOT fixed), and the
    argument texts cross-reference each other by name ("Debater B mistakenly
    identifies ..."). So unlike the legacy swap=True path — which exchanges the
    argument text shown under the A/B labels and would corrupt those
    cross-references — this builder ALWAYS renders the transcript with its
    original labels (swap=False) and expresses the answer ordering only through
    the <ANSWER_A>/<ANSWER_B> texts and the template's
    "<NAME_A> is arguing for answer A" line.

    Callers pass name_a = the debater whose stance equals the option shown as
    answer A (and name_b the other debater). The replace order matches
    build_judge_user_content so prompts stay template-identical otherwise.
    """
    story_normalised = normalize_text(story)
    transcript_str = render_judge_transcript(rounds, story_normalised, swap=False)
    return (
        JUDGE_TEMPLATE
        .replace("<QUESTION>", question)
        .replace("<ANSWER_A>", answer_a)
        .replace("<ANSWER_B>", answer_b)
        .replace("<NAME_A>", name_a)
        .replace("<NAME_B>", name_b)
        .replace("<TRANSCRIPT>", transcript_str)
    )


# ---------------------------------------------------------------------------
# Native (non-harmony) assistant protocol — Llama-3-Instruct student
# ---------------------------------------------------------------------------
# The gpt-oss teacher/judge stack above speaks harmony channels. The Llama-3
# Instruct student has NO analysis/think/final channel at all: one assistant turn
# carries exactly the text the user sees, terminated by <|eot_id|>. These
# constants are the single torch-free source of truth for that protocol, mirroring
# the role FINAL_MARKER/ANALYSIS_MARKER play for harmony. Nothing above this line
# changes: every legacy signature and default keeps its harmony behaviour.
#
# The literal token STRINGS below are protocol, not token ids. Numeric ids are
# ALWAYS derived from the live tokenizer by the callers; this module never
# hardcodes them.
NATIVE_BOS_TOKEN = "<|begin_of_text|>"
NATIVE_EOS_TOKEN = "<|end_of_text|>"
NATIVE_EOT_TOKEN = "<|eot_id|>"
NATIVE_HEADER_START = "<|start_header_id|>"
NATIVE_HEADER_END = "<|end_header_id|>"
NATIVE_HEADER_BODY_SEPARATOR = "\n\n"
NATIVE_ASSISTANT_HEADER = (
    f"{NATIVE_HEADER_START}assistant{NATIVE_HEADER_END}{NATIVE_HEADER_BODY_SEPARATOR}"
)
# The direct readout prefill. Deliberately NOT a control scaffold: it is ordinary
# prose that can also occur inside a user prompt, so NO caller may assume it is
# globally unique in a tokenized input (contrast the harmony scaffold, which is).
NATIVE_ANSWER_PREFILL = "Answer:"
# Separator between a PUBLIC grounded/rationale paragraph and the terminal verdict.
NATIVE_TARGET_SEPARATOR = "\n\n"
NATIVE_CONTROL_TOKENS = (
    NATIVE_BOS_TOKEN,
    NATIVE_EOS_TOKEN,
    NATIVE_EOT_TOKEN,
    NATIVE_HEADER_START,
    NATIVE_HEADER_END,
)
# Stated once, so no downstream report can claim a hidden chain of thought:
NATIVE_NO_PRIVATE_CHANNEL_NOTE = (
    "The native Llama-3 Instruct protocol has no analysis/think/final channel. Any "
    "reasoning text a native student emits is PUBLIC assistant content in the same "
    "turn as the verdict; it is never a private CoT and must not be described as one."
)

# Parser states. Every native parse ends in exactly one of these; RAW_UNKNOWN is a
# refusal, never a silently-salvaged answer.
NATIVE_STATE_FULL_TURN = "FULL_TURN"
NATIVE_STATE_AFTER_HEADER = "AFTER_HEADER"
NATIVE_STATE_AFTER_ANSWER_PREFILL = "AFTER_ANSWER_PREFILL"
NATIVE_STATE_RAW_UNKNOWN = "RAW_UNKNOWN"
NATIVE_PARSER_STATES = (
    NATIVE_STATE_FULL_TURN,
    NATIVE_STATE_AFTER_HEADER,
    NATIVE_STATE_AFTER_ANSWER_PREFILL,
    NATIVE_STATE_RAW_UNKNOWN,
)

# Foreign (harmony) scaffold strings. The native protocol has no channels, but a
# native candidate body is adversarially generated free text, and a literal harmony
# delimiter in it is exactly the injection PLAN section 4 lists as a rejection: the
# tokenizer of a harmony sibling would read it as a control token. Both scaffolds are
# therefore forbidden inside a native assistant body.
NATIVE_FOREIGN_CONTROL_TOKENS = (
    "<|channel|>",
    "<|message|>",
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|constrain|>",
    "<|call|>",
)
NATIVE_BODY_FORBIDDEN_TOKENS = NATIVE_CONTROL_TOKENS + NATIVE_FOREIGN_CONTROL_TOKENS

# The ONLY accepted verdict shape for a whole assistant body: an optional PUBLIC
# analysis, the fixed blank-line separator, then the literal "Answer:", exactly ONE
# ASCII space, one letter, and nothing else. Applied with fullmatch, so "Final
# Answer: A", "Answer: <A>", "Answer:\tA", "Answer:  A", "Answer:A", " Answer: A",
# "Answer: A " and an "Answer: A" buried in prose all fail to parse rather than being
# salvaged.
_NATIVE_VERDICT_RE = re.compile(
    r"(?:(?P<public_g>.+)"
    + re.escape(NATIVE_TARGET_SEPARATOR)
    + r")?"
    + re.escape(NATIVE_ANSWER_PREFILL)
    + r" (?P<letter>[AB])",
    re.S,
)
# A second verdict-shaped line anywhere inside the PUBLIC analysis is two answers,
# which is a refusal rather than a majority vote.
_NATIVE_VERDICT_LINE_RE = re.compile(
    r"^" + re.escape(NATIVE_ANSWER_PREFILL) + r" [AB]$", re.M
)
# The ONLY accepted AFTER_ANSWER_PREFILL completion: one ASCII space, one letter.
_NATIVE_PREFILL_COMPLETION_RE = re.compile(r" (?P<letter>[AB])")


def _native_refusal(reason, state=NATIVE_STATE_RAW_UNKNOWN):
    return {
        "state": state,
        "body": "",
        "public_analysis": None,
        "letter": None,
        "eot_observed": False,
        "terminal": None,
        "reason": reason,
    }


def parse_native_assistant_turn(decoded, *, prompt=None, prefill=None):
    """STRICT native-protocol parse of exactly ONE assistant turn. Never raises.

    Returns {"state", "body", "public_analysis", "letter", "eot_observed",
    "terminal", "reason"}. ``letter`` is non-None ONLY when every clause holds:

      * the input is one of exactly three legal shapes, with no discarded prefix or
        history — FULL_TURN (``prompt``/``prefill`` both None) must BEGIN with the
        assistant header after at most one BOS; AFTER_HEADER (``prefill=""``) is a
        continuation of the generation prompt; AFTER_ANSWER_PREFILL (``prefill``
        exactly "Answer:") is a continuation of the direct readout. Anything else,
        including RAW_UNKNOWN text, a second BOS, preceding prose or an earlier user
        turn, is refused;
      * the terminal <|eot_id|> was actually OBSERVED as the FIRST stop token, so a
        run truncated at max_new_tokens, or one stopped by <|end_of_text|>, can never
        be scored as a verdict. There is no opt-out keyword: a caller cannot ask for a
        letter without an observed terminator;
      * NOTHING follows that terminal — not a newline, not <|end_of_text|>, not
        another turn. Trailing bytes are a refusal, never something we discard;
      * no native or harmony control token appears inside the body (prompt-injected
        scaffold in adversarially generated text);
      * the body is exactly ``[PUBLIC analysis + "\\n\\n"] + "Answer: X"``, the same
        contract the native continuation serializer writes: a single ASCII space, one
        letter, one verdict, the fixed blank-line separator when PUBLIC analysis is
        present, and no second verdict line inside that analysis. In
        AFTER_ANSWER_PREFILL the generated text must be exactly " A"/" B".

    This is additive: the harmony parsers (predicted_letter / extract_*_channel) are
    untouched and remain the default for gpt-oss.
    """
    if not isinstance(decoded, str):
        return _native_refusal(f"expected str, got {type(decoded).__name__}")
    if prefill is not None and not isinstance(prefill, str):
        return _native_refusal(f"prefill must be str, got {type(prefill).__name__}")

    rest = decoded
    if prompt is not None:
        if not isinstance(prompt, str):
            return _native_refusal(f"prompt must be str, got {type(prompt).__name__}")
        if not rest.startswith(prompt):
            return _native_refusal("decoded text does not start with the supplied prompt")
        rest = rest[len(prompt):]
        if prompt.endswith(NATIVE_ASSISTANT_HEADER):
            implied = ""
        elif prompt.endswith(NATIVE_ASSISTANT_HEADER + NATIVE_ANSWER_PREFILL):
            implied = NATIVE_ANSWER_PREFILL
        else:
            return _native_refusal(
                "the supplied prompt does not end at the native assistant header "
                f"{NATIVE_ASSISTANT_HEADER!r} (optionally followed by exactly "
                f"{NATIVE_ANSWER_PREFILL!r}); there is no legal continuation point"
            )
        if prefill is None:
            prefill = implied
        elif prefill != implied:
            return _native_refusal(
                f"supplied prefill {prefill!r} contradicts the prompt tail {implied!r}"
            )

    if prefill is None:
        state = NATIVE_STATE_FULL_TURN
        head = rest[len(NATIVE_BOS_TOKEN):] if rest.startswith(NATIVE_BOS_TOKEN) else rest
        if not head.startswith(NATIVE_ASSISTANT_HEADER):
            return _native_refusal(
                f"a full-turn parse must BEGIN with {NATIVE_ASSISTANT_HEADER!r} after at "
                "most one BOS; refusing to discard a prefix (preceding prose, an earlier "
                "user/assistant turn, or a repeated BOS). Pass prompt=/prefill= to parse "
                "a continuation of a known prompt instead."
            )
        rest = head[len(NATIVE_ASSISTANT_HEADER):]
        prefix_text = ""
    elif prefill == "":
        state = NATIVE_STATE_AFTER_HEADER
        prefix_text = ""
    elif prefill == NATIVE_ANSWER_PREFILL:
        state = NATIVE_STATE_AFTER_ANSWER_PREFILL
        prefix_text = NATIVE_ANSWER_PREFILL
    else:
        return _native_refusal(
            f"unrecognised prefill {prefill!r}: the only legal continuation points are "
            f"'' (the generation prompt) and exactly {NATIVE_ANSWER_PREFILL!r} (the "
            "direct readout prefill)"
        )

    terminal, cut = None, len(rest)
    for token in (NATIVE_EOT_TOKEN, NATIVE_EOS_TOKEN):
        k = rest.find(token)
        if k != -1 and k < cut:
            cut, terminal = k, token
    generated = rest[:cut]
    result = {
        "state": state,
        "body": prefix_text + generated,
        "public_analysis": None,
        "letter": None,
        "eot_observed": terminal == NATIVE_EOT_TOKEN,
        "terminal": terminal,
        "reason": None,
    }

    def refuse(reason):
        result["reason"] = reason
        return result

    if terminal is None:
        return refuse(
            f"no terminal {NATIVE_EOT_TOKEN} was observed; the turn is truncated or "
            "unterminated and must not be scored as a verdict"
        )
    if terminal != NATIVE_EOT_TOKEN:
        return refuse(
            f"the first stop token is {terminal!r}, not the required turn terminator "
            f"{NATIVE_EOT_TOKEN}"
        )
    trailing = rest[cut + len(terminal):]
    if trailing != "":
        return refuse(
            f"{len(trailing)} byte(s) follow the terminal {NATIVE_EOT_TOKEN} "
            f"({trailing[:32]!r}): an extra turn, stop token or whitespace would have to "
            "be discarded"
        )
    for token in NATIVE_BODY_FORBIDDEN_TOKENS:
        if token in result["body"]:
            return refuse(
                f"control token {token!r} appears inside the assistant body (injected "
                "turn boundary or foreign harmony scaffold)"
            )

    if state == NATIVE_STATE_AFTER_ANSWER_PREFILL:
        match = _NATIVE_PREFILL_COMPLETION_RE.fullmatch(generated)
        if match is None:
            return refuse(
                f"after the {NATIVE_ANSWER_PREFILL!r} prefill the completion must be "
                f"exactly one ASCII space and one letter, got {generated!r}"
            )
        result["letter"] = match.group("letter")
        return result

    match = _NATIVE_VERDICT_RE.fullmatch(result["body"])
    if match is None:
        return refuse(
            "body is not exactly [PUBLIC analysis + separator] + "
            f"{NATIVE_ANSWER_PREFILL!r} + ' ' + letter: {result['body'][-48:]!r}"
        )
    public_g = match.group("public_g")
    if public_g is not None:
        if not public_g.strip():
            return refuse("the PUBLIC analysis before the verdict is blank")
        if public_g.endswith("\n"):
            return refuse(
                "the separator before the verdict is not exactly the fixed "
                f"{NATIVE_TARGET_SEPARATOR!r} blank line"
            )
        if _NATIVE_VERDICT_LINE_RE.search(public_g):
            return refuse(
                "a second verdict line appears inside the PUBLIC analysis; multiple "
                "answers are a refusal, not a majority vote"
            )
        result["public_analysis"] = public_g
    result["letter"] = match.group("letter")
    return result


def native_predicted_letter(decoded, *, prompt=None, prefill=None):
    """Strict native counterpart of predicted_letter: 'A'/'B', or None on ANY contract
    violation (unknown shape, discarded prefix, missing/late terminal EOT, trailing
    bytes, injected control token, non-canonical Answer syntax, two answers, a body
    after the direct-readout prefill)."""
    return parse_native_assistant_turn(decoded, prompt=prompt, prefill=prefill)["letter"]
