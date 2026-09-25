"""
Modified as implement Qwen3.5-9B
"""

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
def find_answer(text, letter, *, model_family="gpt-oss", prefix=None):
    if model_family != "gpt-oss":
        return letter in ("A", "B") and predicted_letter(text, model_family=model_family, prefix=prefix) == letter
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


def predicted_letter(judgement, *, model_family="gpt-oss", prefix=None):
    """Return the judge's chosen answer as 'A' / 'B', or None if inconclusive/unparseable.

    A None result counts as the judge being wrong (the reference scores inconclusive as
    incorrect rather than dropping it).
    """
    if model_family != "gpt-oss":
        final = extract_final_channel(judgement, model_family=model_family, prefix=prefix)
        match = re.fullmatch(r"(?:(?:Final )?Answer: )?(?:([AB])|<([AB])>)", final)
        return (match.group(1) or match.group(2)) if match else None
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


def extract_final_channel(decoded, *, model_family="gpt-oss", prefix=None):
    """Pull the harmony `final` channel out of a raw generated string.

    Matches what vLLM's `message.content` gave us: the user-facing answer, with the
    `analysis` CoT channel dropped. We take the text after the LAST final-channel
    marker (a turn can contain analysis then final), trimmed at the stop token. If
    no final marker is present (e.g. generation hit max_new_tokens mid-analysis),
    return "" so it scores as inconclusive — we never feed raw analysis CoT to the
    answer parser, since the old vLLM message.content was final-channel only.
    """
    if model_family != "gpt-oss":
        if model_family == "qwen3_5":
            return extract_qwen_final_channel(decoded, prefix=prefix)
        _require_gemma_family(model_family)
        return extract_gemma_final_channel(decoded, prefix=prefix)
    idx = decoded.rfind(FINAL_MARKER)
    if idx == -1:
        return ""
    text = decoded[idx + len(FINAL_MARKER):]
    for stop in ("<|return|>", "<|end|>"):
        k = text.find(stop)
        if k != -1:
            text = text[:k]
    return text.strip()


def extract_analysis_channel(decoded, *, model_family="gpt-oss", prefix=None):
    """Pull the harmony `analysis` (CoT) channel out of a raw generated string.

    Mirror of extract_final_channel for the analysis channel. We take the text after
    the FIRST analysis-channel marker (analysis precedes final within a turn), trimmed
    at the next channel/stop token. Returns "" if no analysis marker is present. This
    is a DIAGNOSTIC helper only (e.g. checking whether a fine-tuned verifier's analysis
    text is non-constant across examples); like the CoT itself it is never fed to the
    answer parser.
    """
    if model_family != "gpt-oss":
        if model_family == "qwen3_5":
            return extract_qwen_analysis_channel(decoded, prefix=prefix)
        _require_gemma_family(model_family)
        return extract_gemma_analysis_channel(decoded, prefix=prefix)
    idx = decoded.find(ANALYSIS_MARKER)
    if idx == -1:
        return ""
    text = decoded[idx + len(ANALYSIS_MARKER):]
    for stop in ("<|end|>", "<|start|>", "<|channel|>", "<|return|>"):
        k = text.find(stop)
        if k != -1:
            text = text[:k]
    return text.strip()


# Gemma constants stay separate from the legacy Harmony public constants. They
# intentionally duplicate immutable ft-verifier.py's StudentProtocol; exact real
# q_m/q_h readout tests lock parity without monkeypatching either module.
GEMMA_PROTOCOL_VERSION = "gemma4-thought-direct-v1"
GEMMA_TEMPLATE_VERSION = "gemma4-thinking-common-base-v1"
GEMMA_GENERATION_SUFFIX = "<|turn>model\n"
GEMMA_ANALYSIS_PREFIX = "<|channel>thought\n"
GEMMA_DIRECT_PREFILL = "<|channel>thought\n<channel|>Answer:"
GEMMA_CONTROL_TOKENS = ("<|channel>", "<channel|>", "<|turn>", "<turn|>")


def _require_gemma_family(family):
    if family != "gemma4":
        raise ValueError(f"unsupported model_family {family!r}; use 'gpt-oss' or 'gemma4'")


def gemma_native_control_ids(tokenizer):
    """Detect canonical native controls (including the turn stop, NOT EOS)."""
    controls = {}
    for token in GEMMA_CONTROL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        expected = tokenizer.convert_tokens_to_ids(token)
        special = (token in tokenizer.all_special_tokens or bool(getattr(
            getattr(tokenizer, "added_tokens_decoder", {}).get(expected), "special", False)))
        if len(ids) != 1 or ids[0] != expected or not special or expected == tokenizer.unk_token_id:
            raise RuntimeError(f"Gemma control token {token!r} is not canonical: {ids!r}")
        controls[token] = int(expected)
    if len(set(controls.values())) != len(controls):
        raise RuntimeError("Gemma native control IDs are not distinct")
    if controls["<turn|>"] == tokenizer.eos_token_id:
        raise RuntimeError("Gemma native turn stop must remain distinct from tokenizer EOS")
    return controls


def _gemma_ids(encoded):
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (tuple, list)):
        if len(encoded) != 1:
            raise RuntimeError("expected one Gemma conversation")
        encoded = encoded[0]
    return [int(i) for i in encoded]


def gemma_readout_ids(tokenizer, messages):
    """Return (readout IDs, prefill IDs, letter IDs, control IDs), torch-free.

    Only system/user semantic messages are accepted. Verify text, native template,
    standalone prefill and contextual A/B boundaries; never generate reasoning.
    """
    if not messages or any(m.get("role") not in {"system", "user"} for m in messages):
        raise RuntimeError("Gemma base prompt requires non-empty system/user messages only")
    for message in messages:
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemma prompt content must be non-empty text")
        if re.search(r"<\||\|>", text) or any(t and t in text for t in tokenizer.all_special_tokens):
            raise RuntimeError("Gemma prompt contains a native control delimiter/special token")
    kwargs = dict(add_generation_prompt=True, enable_thinking=True)
    base = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if (not base.endswith(GEMMA_GENERATION_SUFFIX) or base.count("<|think|>") != 1
            or "<|channel>" in base or "<channel|>" in base):
        raise RuntimeError("Gemma template requires one <|think|>, unopened thought, and native model suffix")
    # Validate turn boundaries as well as the final suffix; a template that drops
    # a user stop or embeds another control must not silently pass the probe.
    head = base[:-len(GEMMA_GENERATION_SUFFIX)]
    bos = tokenizer.bos_token
    if not bos or not head.startswith(bos):
        raise RuntimeError("Gemma native template must start with BOS")
    head = head[len(bos):]
    turns = list(re.finditer(r"<\|turn>(system|user)\n(.*?)<turn\|>\n", head, re.DOTALL))
    if not turns or "".join(t.group(0) for t in turns) != head:
        raise RuntimeError("Gemma native template has malformed system/user turn boundaries")
    for turn in turns:
        content = turn.group(2)
        if turn.group(1) == "system":
            content = content.replace("<|think|>", "")
        if re.search(r"<\||\|>", content) or any(t and t in content for t in tokenizer.all_special_tokens):
            raise RuntimeError("Gemma native template contains unexpected control tokens")
    base_ids = _gemma_ids(tokenizer.apply_chat_template(messages, tokenize=True, **kwargs))
    encode = lambda text: _gemma_ids(tokenizer(text, add_special_tokens=False))
    if not base_ids or base_ids != encode(base):
        raise RuntimeError("Gemma template text and tokenized output disagree")
    controls = gemma_native_control_ids(tokenizer)
    prefill = encode(GEMMA_DIRECT_PREFILL)
    if ([i for i in prefill if i in controls.values()]
            != [controls["<|channel>"], controls["<channel|>"]]):
        raise RuntimeError("Gemma direct prefill lost its native channel boundaries")
    readout = base_ids + prefill
    if encode(base + GEMMA_DIRECT_PREFILL) != readout:
        raise RuntimeError("Gemma base/prefill token boundary mismatch")
    letters = {}
    for letter in ("A", "B"):
        ids = encode(" " + letter)
        if len(ids) != 1 or ids[0] in controls.values() or ids[0] == tokenizer.unk_token_id:
            raise RuntimeError(f"Gemma completion {letter!r} must be one canonical letter token")
        letters[letter] = ids[0]
        continuation = GEMMA_DIRECT_PREFILL + " " + letter + "<turn|>"
        tail = prefill + ids + [controls["<turn|>"]]
        if (encode(continuation) != tail or encode(base + continuation) != base_ids + tail
                or encode(base + GEMMA_DIRECT_PREFILL + " " + letter) != readout + ids):
            raise RuntimeError("Gemma contextual prefill/letter/native-stop boundary mismatch")
    if letters["A"] == letters["B"]:
        raise RuntimeError("Gemma A/B letter IDs collide")
    return readout, prefill, letters, controls


def _gemma_channels(decoded, prefix=None):
    """Strict native continuation grammar; returns (final, diagnostic analysis).

    prefix=None: require a native model turn or an explicit thought opener (a full
    native conversation is also accepted). For NEW-TOKEN-ONLY decoding use:
      'model'   prompt ends at <|turn>model\n; thought must arrive in decoded.
      'thought' prompt already ends at <|channel>thought\n; require its close.
      'final'   prompt already contains the closed, empty thought; raw final follows.
      'answer'  prompt already contains GEMMA_DIRECT_PREFILL; decoded starts ' A/B'.
    These are caller assertions about the prompt, not auto-detection heuristics.
    No prompt/user text need be decoded. A newer user turn or unfinished model turn
    invalidates a prior final. Unknown/duplicate/mixed/partial controls fail closed.
    A structurally valid final may lack a stop (max-token truncation); open thought
    never supplies an answer. Analysis is diagnostic ONLY.
    """
    contexts = {None: "", "model": GEMMA_GENERATION_SUFFIX,
                "thought": GEMMA_GENERATION_SUFFIX + GEMMA_ANALYSIS_PREFIX,
                "final": GEMMA_GENERATION_SUFFIX + GEMMA_ANALYSIS_PREFIX + "<channel|>",
                "answer": GEMMA_GENERATION_SUFFIX + GEMMA_DIRECT_PREFILL}
    if prefix not in contexts:
        raise ValueError("Gemma prefix must be None, 'model', 'thought', 'final', or 'answer'")
    if not isinstance(decoded, str):
        return "", ""
    text = contexts[prefix] + decoded
    if prefix is None:
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]
        if text.startswith(GEMMA_ANALYSIS_PREFIX):
            text = GEMMA_GENERATION_SUFFIX + text
    final = analysis = ""
    while text:
        header = re.match(r"<\|turn>(system|user|model)\n", text)
        if not header:
            return "", ""
        role = header.group(1)
        body = text[header.end():]
        boundary = body.find("<|turn>")
        rest = "" if boundary == -1 else body[boundary:]
        body = body if boundary == -1 else body[:boundary]
        # Only one native turn stop and optional terminal EOS. EOS cannot bridge turns.
        stopped = False
        if body.rstrip().endswith("<eos>"):
            if rest:
                return "", ""
            body = body.rstrip()[:-len("<eos>")]
            stopped = True
        if body.rstrip().endswith("<turn|>"):
            body = body.rstrip()[:-len("<turn|>")]
            stopped = True
        if rest and not stopped:
            return "", ""
        final = analysis = ""
        if role != "model":
            if not stopped:
                return "", ""
            if role == "system" and body.count("<|think|>") <= 1:
                body = body.replace("<|think|>", "")
            if re.search(r"<\||\|>|<bos>|<eos>|<pad>", body):
                return "", ""
        else:
            if not body.startswith(GEMMA_ANALYSIS_PREFIX):
                # The native chat template strips thoughts from historical model
                # turns, leaving visible content only. Skip a completed past turn,
                # but never use it as evidence for the current answer or accept an
                # unstructured latest continuation without an explicit prefix.
                if rest and stopped and not re.search(r"<\||\|>|<bos>|<eos>|<pad>", body):
                    text = rest
                    continue
                return "", ""
            payload = body[len(GEMMA_ANALYSIS_PREFIX):]
            parts = payload.split("<channel|>")
            if len(parts) > 2 or any(re.search(r"<\||\|>|<bos>|<eos>|<pad>", p) for p in parts):
                return "", ""
            if len(parts) == 1 and rest:
                return "", ""  # an unclosed thought cannot be repaired by a later turn
            analysis = parts[0].strip()
            if len(parts) == 2:
                final = parts[1].strip()
        text = rest
    return final, analysis


def extract_gemma_final_channel(decoded, *, prefix=None):
    return _gemma_channels(decoded, prefix)[0]


def extract_gemma_analysis_channel(decoded, *, prefix=None):
    return _gemma_channels(decoded, prefix)[1]


# ---- Qwen3.5 native protocol (OPT-IN; the default parser stays Harmony) -----
# These constants intentionally duplicate immutable ft-verifier.py's QWEN_PROTOCOL
# StudentProtocol, exactly as the Gemma block above does: exact real q_m/q_h readout
# parity tests lock them together without either module importing the other.
QWEN_PROTOCOL_VERSION = "qwen35-think-direct-v1"
QWEN_TEMPLATE_VERSION = "qwen35-thinking-common-base-v1"
QWEN_GENERATION_SUFFIX = "<|im_start|>assistant\n"
# The native thinking template OPENS the model's think block after the assistant
# header. The COMMON base strips exactly this opener so the readout owns the whole
# block; leaving it in breaks token-prefix identity (the template newline BPE-merges
# with the continuation newline).
QWEN_NATIVE_OPENER = "<think>\n"
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"
QWEN_EMPTY_THINK = "<think>\n\n</think>\n\n"
QWEN_DIRECT_PREFILL = "<think>\n\n</think>\n\nAnswer:"
QWEN_TURN_STOP = "<|im_end|>"
QWEN_CONTROL_TOKENS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
# Which controls must be REGISTERED special. Qwen's think tags are canonical regular
# ADDED tokens on this release, so demanding special=True would be a false requirement;
# they are validated by the single-ID/convert/round-trip checks instead.
QWEN_STRICT_SPECIAL_CONTROLS = ("<|im_start|>", "<|im_end|>")


def _qwen_residue(text):
    """Any native delimiter, or a complete/partial/unknown think tag, in free text.

    For PARSED MODEL OUTPUT only. Prompt content uses _qwen_prompt_control_hit, which
    is deliberately the looser trainer rule; see there.
    """
    return bool(re.search(r"<\||\|>|</?think", text))


def _qwen_prompt_control_hit(text, specials=()):
    """Native control injection in PROMPT content, using the trainer's exact rule.

    ft-verifier.assert_student_content rejects the <| |> delimiter plus the exact
    control and special tokens, and nothing else. Real verifier prompts legitimately
    instruct the debater to reason in <thinking></thinking> tags, so the looser
    output-side residue regex (which matches any '<think' prefix) would reject the
    actual training corpus and break q_m/q_h readout parity. Partial tags are harmless
    here: only the exact literal maps to a native control ID.
    """
    return bool(re.search(r"<\||\|>", text)
                or any(t in text for t in QWEN_CONTROL_TOKENS)
                or any(t and t in text for t in specials))


def qwen_native_control_ids(tokenizer):
    """Canonical distinct native control IDs (turn stop included; it IS this
    tokenizer's EOS, unlike Gemma, so no distinctness claim is made about EOS)."""
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    controls = {}
    for token in QWEN_CONTROL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        expected = tokenizer.convert_tokens_to_ids(token)
        registered_special = (token in tokenizer.all_special_tokens
                              or bool(getattr(added.get(expected), "special", False)))
        if (len(ids) != 1 or expected is None or ids[0] != expected
                or expected == tokenizer.unk_token_id
                or (token in QWEN_STRICT_SPECIAL_CONTROLS and not registered_special)):
            raise RuntimeError(f"Qwen control token {token!r} is not canonical: "
                               f"encode={ids}, convert={expected!r}")
        if token not in QWEN_STRICT_SPECIAL_CONTROLS:
            # Structural regular added token: still a real, single, distinct,
            # self-decoding unit, not silently re-segmented ordinary text.
            if (tokenizer.convert_ids_to_tokens(expected) != token
                    or tokenizer.decode([expected]) != token
                    or expected not in added):
                raise RuntimeError(f"Qwen control token {token!r} is not a canonical "
                                   f"regular added token (id={expected!r})")
        controls[token] = int(expected)
    if len(set(controls.values())) != len(controls):
        raise RuntimeError(f"Qwen native control IDs are not distinct: {controls!r}")
    return controls


def _qwen_ids(encoded):
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (tuple, list)):
        if len(encoded) != 1:
            raise RuntimeError("expected one Qwen conversation")
        encoded = encoded[0]
    return [int(i) for i in encoded]


def _qwen_occurrences(haystack, needle):
    n = len(needle)
    return sum(1 for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle)


def qwen_readout_ids(tokenizer, messages):
    """Return (readout IDs, prefill IDs, letter IDs, control IDs), torch-free.

    Only system/user semantic messages are accepted. Verifies text, the native
    template's actual turn structure, the token-exact opener strip, the standalone
    prefill and the whole base+prefill+letter+native-stop contextual boundaries;
    never generates reasoning. The tokenizer-vs-model embedding range is the
    caller's check (only it holds the config).
    """
    if not messages or any(m.get("role") not in {"system", "user"} for m in messages):
        raise RuntimeError("Qwen base prompt requires non-empty system/user messages only")
    specials = tuple(t for t in tokenizer.all_special_tokens if t)
    for message in messages:
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Qwen prompt content must be non-empty text")
        # Qwen's think tags carry no <| |> delimiter, so the delimiter regex alone
        # cannot see them; the control/special scan is additive, as in ft-verifier.py.
        if _qwen_prompt_control_hit(text, specials):
            raise RuntimeError("Qwen prompt contains a native control delimiter/special token")
    controls = qwen_native_control_ids(tokenizer)
    kwargs = dict(add_generation_prompt=True, enable_thinking=True)
    native = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not native.endswith(QWEN_GENERATION_SUFFIX + QWEN_NATIVE_OPENER):
        raise RuntimeError("Qwen thinking template must end with the assistant header "
                           f"plus {QWEN_NATIVE_OPENER!r}; got {native[-120:]!r}")
    base = native[:-len(QWEN_NATIVE_OPENER)]
    if (not base.endswith(QWEN_GENERATION_SUFFIX)
            or QWEN_THINK_OPEN in base or QWEN_THINK_CLOSE in base):
        raise RuntimeError("Qwen common base must end at the bare assistant header "
                           "with no think block once the native opener is stripped")
    # Validate actual turn boundaries as well as the final suffix; a template that
    # drops a stop or embeds another control must not silently pass the probe.
    head = base[:-len(QWEN_GENERATION_SUFFIX)]
    bos = tokenizer.bos_token
    if bos and head.startswith(bos):
        raise RuntimeError("Qwen native template must not emit BOS")
    turns = list(re.finditer(r"<\|im_start\|>(system|user)\n(.*?)<\|im_end\|>\n", head, re.DOTALL))
    if not turns or "".join(t.group(0) for t in turns) != head:
        raise RuntimeError("Qwen native template has malformed system/user turn boundaries")
    for turn in turns:
        content = turn.group(2)
        if _qwen_prompt_control_hit(content, specials):
            raise RuntimeError("Qwen native template contains unexpected control tokens")
    encode = lambda text: _qwen_ids(tokenizer(text, add_special_tokens=False))
    native_ids = _qwen_ids(tokenizer.apply_chat_template(messages, tokenize=True, **kwargs))
    base_ids = encode(base)
    opener_ids = encode(QWEN_NATIVE_OPENER)
    if not base_ids or not opener_ids or native_ids != base_ids + opener_ids:
        raise RuntimeError("stripping the Qwen native generation opener is not token-exact")
    prefill = encode(QWEN_DIRECT_PREFILL)
    if (not prefill or prefill[0] != controls[QWEN_THINK_OPEN]
            or controls[QWEN_THINK_CLOSE] not in prefill):
        raise RuntimeError("Qwen direct prefill lost its native think boundaries")
    readout = base_ids + prefill
    if encode(base + QWEN_DIRECT_PREFILL) != readout:
        raise RuntimeError("Qwen base/prefill token boundary mismatch")
    if (readout[-len(prefill):] != prefill
            or _qwen_occurrences(readout, prefill) != 1):
        raise RuntimeError("the Qwen direct prefill must be the readout suffix exactly once")
    letters = {}
    for letter in ("A", "B"):
        ids = encode(" " + letter)
        if len(ids) != 1 or ids[0] in controls.values() or ids[0] == tokenizer.unk_token_id:
            raise RuntimeError(f"Qwen completion {letter!r} must be one canonical letter token")
        letters[letter] = ids[0]
        continuation = QWEN_DIRECT_PREFILL + " " + letter + QWEN_TURN_STOP
        tail = prefill + ids + [controls[QWEN_TURN_STOP]]
        if (encode(continuation) != tail or encode(base + continuation) != base_ids + tail
                or encode(base + QWEN_DIRECT_PREFILL + " " + letter) != readout + ids):
            raise RuntimeError("Qwen contextual prefill/letter/native-stop boundary mismatch")
        if readout[-1] == ids[0]:
            raise RuntimeError("an answer letter is already appended to the Qwen readout "
                               "input; the letter is read from the logits, never forced")
    if letters["A"] == letters["B"]:
        raise RuntimeError("Qwen A/B letter IDs collide")
    return readout, prefill, letters, controls


def _qwen_channels(decoded, prefix=None):
    """Strict native continuation grammar; returns (final, diagnostic analysis).

    prefix=None: a full native conversation. For NEW-TOKEN-ONLY decoding use:
      'assistant' prompt ends at the BARE <|im_start|>assistant\n header. It asserts
                  nothing else, so the think block must OPEN and CLOSE inside decoded;
                  bare prose is rejected exactly as with prefix=None.
      'think'     prompt already ends at the native opener; require its close.
      'final'     prompt already contains the closed, EMPTY think; raw final follows.
      'answer'    prompt already contains QWEN_DIRECT_PREFILL; decoded starts ' A/B'.
    These are caller assertions about the prompt, not auto-detection heuristics. No
    prompt/user text need be decoded. A newer user turn or an unfinished assistant
    turn invalidates a prior final, and under a caller prefix the decoded text is a
    single generated turn: opening any further turn is an overrun and fails closed. Unknown/duplicate/mixed/partial controls fail
    closed. A structurally valid final may lack a stop (max-token truncation); an
    open think NEVER supplies an answer and a later turn can never repair it. The
    native template strips reasoning from historical assistant turns, so a COMPLETED
    plain past turn is skipped rather than trusted. Analysis is diagnostic ONLY.
    """
    contexts = {None: "", "assistant": QWEN_GENERATION_SUFFIX,
                "think": QWEN_GENERATION_SUFFIX + QWEN_NATIVE_OPENER,
                "final": QWEN_GENERATION_SUFFIX + QWEN_EMPTY_THINK,
                "answer": QWEN_GENERATION_SUFFIX + QWEN_DIRECT_PREFILL}
    if prefix not in contexts:
        raise ValueError("Qwen prefix must be None, 'assistant', 'think', 'final', or 'answer'")
    if not isinstance(decoded, str):
        return "", ""
    text = contexts[prefix] + decoded
    final = analysis = ""
    while text:
        header = re.match(r"<\|im_start\|>(system|user|assistant)\n", text)
        if not header:
            return "", ""
        role = header.group(1)
        body = text[header.end():]
        boundary = body.find("<|im_start|>")
        rest = "" if boundary == -1 else body[boundary:]
        body = body if boundary == -1 else body[:boundary]
        # A caller prefix asserts ONE prompt-completing assistant turn. New tokens that
        # go on to open a further turn ran past the native stop, so the asserted turn's
        # answer cannot be read back out of a continuation nobody asserted; fail closed.
        if rest and prefix is not None:
            return "", ""
        # Exactly one terminal native turn stop; it can never bridge two turns.
        stopped = False
        stripped = body.rstrip("\n")
        if stripped.endswith(QWEN_TURN_STOP):
            body = stripped[:-len(QWEN_TURN_STOP)]
            stopped = True
        if (rest and not stopped) or QWEN_TURN_STOP in body:
            return "", ""
        final = analysis = ""
        if role != "assistant":
            if not stopped or _qwen_residue(body):
                return "", ""
        elif not body.startswith(QWEN_THINK_OPEN):
            # A completed historical assistant turn legitimately carries visible
            # content only. Skip it, but never use it as evidence for the current
            # answer, and never accept an unstructured latest continuation.
            if not (rest and stopped and not _qwen_residue(body)):
                return "", ""
        else:
            payload = body[len(QWEN_THINK_OPEN):]
            parts = payload.split(QWEN_THINK_CLOSE)
            if len(parts) > 2 or any(_qwen_residue(part) for part in parts):
                return "", ""  # repeated/malformed/partial think controls
            if len(parts) == 1 and rest:
                return "", ""  # an unclosed think cannot be repaired by a later turn
            analysis = parts[0].strip()
            if len(parts) == 2:
                final = parts[1].strip()
        text = rest
    return final, analysis


def extract_qwen_final_channel(decoded, *, prefix=None):
    return _qwen_channels(decoded, prefix)[0]


def extract_qwen_analysis_channel(decoded, *, prefix=None):
    return _qwen_channels(decoded, prefix)[1]



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
