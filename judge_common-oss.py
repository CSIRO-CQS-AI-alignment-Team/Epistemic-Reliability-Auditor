"""Shared judge/verifier prompt rendering and Harmony-channel parsing.

The module is torch-free so dataset construction and audits can run without loading a
model stack. ``ft-verifier-oss-full.py`` and the forced-choice scorers reuse the same
configured prompt, debater-to-answer mapping, quote verification, and final-channel
parser.

Configuration is loaded from ``config/judge-default.yaml`` or the path supplied through
``JUDGE_CONFIG``. Paths are resolved independently of the caller's working directory,
and a missing configuration fails explicitly.
"""

import os
import re

import yaml

# Quote verification and normalization are shared through the stdlib-only helper,
# keeping this module torch-free and import-light.
from quote_utils import normalize_text, verify_quotes

# ---- Shared prompt configuration -------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("JUDGE_CONFIG") or os.path.join(_HERE, "config", "judge-default.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Judge config not found at {CONFIG_PATH!r}. Restore the tracked "
        "config/judge-default.yaml file or point JUDGE_CONFIG at a valid judge config."
    )

CONFIG = yaml.safe_load(open(CONFIG_PATH))
LM_CONFIG = CONFIG["language_model"]
JUDGE_TEMPLATE = CONFIG["prompts"]["messages"][0]["content"]
JUDGE_MAX_TOKENS = 2048

# gpt-oss emits an ``analysis`` channel followed by a ``final`` channel. Answer
# parsing consumes only the final channel.
FINAL_MARKER = "<|channel|>final<|message|>"
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"


# ---- Answer parsing and scoring --------------------------------------------
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

    Return the text after the last final-channel marker, trimmed at the stop token.
    If generation ends before a final channel appears, return ``""`` so the output is
    scored as inconclusive and analysis text never reaches the answer parser.
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
    identifies ..."). This builder therefore always renders the transcript with its
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
