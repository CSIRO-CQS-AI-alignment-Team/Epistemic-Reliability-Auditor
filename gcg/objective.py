"""Discrete GCG objective, task compilation, and suffix-fidelity primitives.

This module implements the parts of GCG that are easy to get subtly wrong:

* every task is a fixed token sequence with one editable token block;
* the suffix is shared across all q_m/q_h tasks in a run;
* the target is the *next token after the final-channel prefill*;
* the loss is restricted to the two production answer tokens, rather than the
  vocabulary-wide language-model loss;
* candidate ranking always uses an exact forward pass after the gradient proposal;
* the token sequence that GCG optimizes must be *exactly* the token sequence that
  the emitted suffix text reproduces when it is re-tokenized in context.

Task-specification contract
---------------------------
The on-disk / wire form is canonical and uses ``"id"`` for the task identifier.
``load_task_spec`` validates and returns those rows **unchanged**; ``RawTask`` is
only an internal view produced by ``_parse_raw_task``. Never write a normalized
row shape back into ``spec["tasks"]`` -- ``compile_task_spec`` re-parses the same
rows and a renamed key silently breaks the whole runner.

Suffix fidelity
---------------
GCG optimizes token ids, but the deliverable is *text* that the production scorer
re-tokenizes. Two tiers keep those in sync (see ``SuffixValidator``):

* tier 1 (cheap): context-free decode/re-encode stability, printability, forbidden
  substrings, plus short boundary probes taken from the real task boundaries;
* tier 2 (exact): re-render ``prompt_prefix + suffix_text + prompt_suffix`` through
  the chat template and require token-for-token equality with the spliced sequence.

Both tiers run on every candidate BEFORE it is scored, so a suffix that could not be
deployed as text never consumes a forward pass; the initial suffix and the final
serialized suffix are checked again. The emitted suffix text therefore always denotes
the token sequence that was scored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


TASK_SCHEMA_VERSION = "gcg-task-v2"
RESULT_SCHEMA_VERSION = "gcg-result-v2"
DEFAULT_PREFILL = "<|channel|>final<|message|>Answer:"
DEFAULT_COMPLETIONS = {"A": " A", "B": " B"}
DEFAULT_EDIT_MARKER = "\n<GCG_EDITABLE_SPAN>\n"
DEFAULT_GENERATION_PROMPT_SUFFIX = "<|start|>assistant"
DEFAULT_SCAFFOLD_CONTROL_TOKENS = ("<|channel|>", "<|message|>")

REPLACEMENT_CHAR = "�"

# Substrings that must never appear inside a public transcript fragment: they would
# either break the `Debater X: """..."""` rendering, forge quote-verification tags,
# or escape the <transcript> block that the verifier prompt wraps around the debate.
FORBIDDEN_SUFFIX_SUBSTRINGS = (
    '"""',
    "<quote>",
    "</quote>",
    "<v_quote>",
    "</v_quote>",
    "<u_quote>",
    "</u_quote>",
    "<transcript>",
    "</transcript>",
)


@dataclasses.dataclass(frozen=True)
class RawTask:
    """A task before it is tokenized with a specific model tokenizer."""

    task_id: str
    group: str
    prompt_prefix: str
    prompt_suffix: str
    target: str
    weight: float = 1.0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CompiledTask:
    """A tokenized task with the editable block removed from the middle."""

    task_id: str
    group: str
    prompt_prefix: str
    prompt_suffix: str
    prefix_ids: torch.Tensor
    tail_ids: torch.Tensor
    prefill_ids: torch.Tensor
    target_index: int
    target_letter: str
    target_token_id: int
    weight: float
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def prefix_length(self) -> int:
        return int(self.prefix_ids.numel())

    @property
    def tail_length(self) -> int:
        return int(self.tail_ids.numel())


@dataclasses.dataclass
class ObjectiveResult:
    """Exact objective and diagnostics for one suffix."""

    loss: float
    per_task_loss: list[float]
    per_task_target_probability: list[float]
    group_loss: dict[str, float]
    group_target_probability: dict[str, float]


def sha256_file(path: str | Path) -> str:
    """Return the full digest of a task specification for provenance."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    return value


def _parse_raw_task(row: Mapping[str, Any], index: int) -> RawTask:
    if not isinstance(row, Mapping):
        raise ValueError(f"tasks[{index}] must be an object")
    task_id = _require_string(row.get("id"), f"tasks[{index}].id")
    group = _require_string(row.get("group", task_id), f"tasks[{index}].group")
    prompt_prefix = _require_string(row.get("prompt_prefix"), f"tasks[{index}].prompt_prefix")
    prompt_suffix = _require_string(row.get("prompt_suffix"), f"tasks[{index}].prompt_suffix")
    target = _require_string(row.get("target"), f"tasks[{index}].target").upper()
    if target not in {"A", "B"}:
        raise ValueError(f"tasks[{index}].target must be 'A' or 'B', got {target!r}")
    try:
        weight = float(row.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tasks[{index}].weight must be finite and positive") from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"tasks[{index}].weight must be finite and positive")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"tasks[{index}].metadata must be an object")
    return RawTask(task_id, group, prompt_prefix, prompt_suffix, target, weight, dict(metadata))


def load_task_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON task specification.

    Every row is validated through ``_parse_raw_task`` but is returned in its
    ORIGINAL wire shape (``"id"``, not ``"task_id"``). ``compile_task_spec`` parses
    the same rows again, so normalizing the key here would break the runner.
    """

    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, Mapping):
        raise ValueError("GCG task specification must be a JSON object")
    version = spec.get("schema_version")
    if version != TASK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported GCG task schema {version!r}; expected {TASK_SCHEMA_VERSION!r}"
        )
    rows = spec.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("GCG task specification must contain a non-empty tasks list")
    seen: set[str] = set()
    preserved: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        task = _parse_raw_task(row, index)
        if task.task_id in seen:
            raise ValueError(f"duplicate GCG task id {task.task_id!r}")
        seen.add(task.task_id)
        preserved.append(dict(row))
    out = dict(spec)
    out["tasks"] = preserved
    out.setdefault("meta", {})
    if not isinstance(out["meta"], Mapping):
        raise ValueError("GCG task specification meta must be an object")
    return out


# Meta keys that change how a spec compiles or deploys. The digest below covers
# these plus every task row, so two specs with the same digest are interchangeable
# for an already-optimized suffix; cosmetic meta (counts, warnings) is excluded.
DIGEST_META_KEYS = (
    "prefill",
    "choice_completions",
    "edit_marker",
    "scaffold_control_tokens",
    "generation_prompt_suffix",
    "suffix_separator",
)


def task_spec_digest(spec: Mapping[str, Any]) -> str:
    """Stable content digest of a task specification.

    Unlike ``sha256_file`` this ignores JSON formatting and key order, so it binds a
    result to the *content* it was optimized against. ``gcg.apply`` recomputes it and
    refuses a result whose digest does not match the spec it is handed.
    """

    meta = spec.get("meta", {})
    meta = meta if isinstance(meta, Mapping) else {}
    payload = {
        "schema_version": spec.get("schema_version"),
        "meta": {key: meta[key] for key in DIGEST_META_KEYS if key in meta},
        "tasks": [
            {
                "id": task.task_id,
                "group": task.group,
                "prompt_prefix": task.prompt_prefix,
                "prompt_suffix": task.prompt_suffix,
                "target": task.target,
                "weight": float(task.weight),
                "metadata": dict(task.metadata),
            }
            for task in parsed_tasks(spec)
        ],
    }
    return sha256_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def parsed_tasks(spec: Mapping[str, Any]) -> list[RawTask]:
    """Internal view of a (raw or loaded) specification as ``RawTask`` objects."""

    rows = spec.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("GCG task specification must contain a non-empty tasks list")
    return [_parse_raw_task(row, index) for index, row in enumerate(rows)]


def _as_id_list(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        if encoded.ndim == 2:
            encoded = encoded[0]
        if encoded.ndim != 1:
            raise ValueError(f"tokenizer returned unexpected ids shape {tuple(encoded.shape)}")
        return [int(x) for x in encoded.tolist()]
    if isinstance(encoded, (list, tuple)):
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(x) for x in encoded]
    raise ValueError(f"tokenizer returned unsupported ids type {type(encoded).__name__}")


def _encode(tokenizer: Any, text: str) -> torch.Tensor:
    """Encode text without silently adding BOS/EOS tokens."""

    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return torch.tensor(_as_id_list(encoded), dtype=torch.long)


def _encode_batch(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """Encode many short strings, preferring one batched tokenizer call."""

    if not texts:
        return []
    try:
        encoded = tokenizer(list(texts), add_special_tokens=False)
        rows = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if isinstance(rows, torch.Tensor):
            rows = rows.tolist()
        if len(rows) == len(texts) and all(isinstance(row, (list, tuple)) for row in rows):
            return [[int(x) for x in row] for row in rows]
    except (TypeError, ValueError, KeyError):
        pass
    return [[int(x) for x in _encode(tokenizer, text).tolist()] for text in texts]


def _decode_batch(tokenizer: Any, id_rows: Sequence[Sequence[int]]) -> list[str]:
    if not id_rows:
        return []
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        try:
            out = batch_decode([list(row) for row in id_rows], skip_special_tokens=False)
            if len(out) == len(id_rows):
                return [str(text) for text in out]
        except (TypeError, ValueError):
            pass
    return [str(tokenizer.decode(list(row), skip_special_tokens=False)) for row in id_rows]


def _chat_ids(tokenizer: Any, content: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return torch.tensor(_as_id_list(encoded), dtype=torch.long)


def _chat_text(tokenizer: Any, content: str) -> str | None:
    """Render the chat template as text, or None if the tokenizer cannot do it."""

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except (TypeError, ValueError):
        return None
    return rendered if isinstance(rendered, str) else None


def _find_contiguous(sequence: torch.Tensor, needle: torch.Tensor) -> list[int]:
    """Diagnostic: locate a token run. Compilation splits by characters instead,
    precisely because a pre-tokenizer merge can erase a marker's token run."""

    if needle.numel() == 0:
        raise ValueError("editable marker tokenization is empty")
    if sequence.numel() < needle.numel():
        return []
    hits = []
    for start in range(int(sequence.numel() - needle.numel()) + 1):
        if torch.equal(sequence[start : start + needle.numel()], needle):
            hits.append(start)
    return hits


def _token_id(tokenizer: Any, completion: str, letter: str) -> int:
    ids = _encode(tokenizer, completion)
    if ids.numel() != 1:
        raise ValueError(
            f"forced-choice completion {completion!r} for {letter} must be exactly one "
            f"token, got ids={ids.tolist()}; use a tokenizer/model with a one-token "
            "answer slot or provide a compatible completion mapping"
        )
    return int(ids[0])


def _require_production_scaffold(
    prefill_text: str,
    control_tokens: Sequence[str],
    generation_prompt_suffix: Any,
) -> None:
    """The scored slot is the production answer slot or the run does not happen.

    This pipeline optimizes the production forced-choice readout at
    ``<|channel|>final<|message|>Answer:``. Anything else -- an analysis-channel
    prefill, a truncated scaffold, trailing junk after ``Answer:``, or task meta that
    blanks the guards out -- would score a slot the production verifier never reads,
    so the values are pinned rather than merely pattern-checked.
    """

    if prefill_text != DEFAULT_PREFILL:
        raise ValueError(
            f"final-channel prefill must be exactly {DEFAULT_PREFILL!r} to match the "
            f"production forced-choice answer slot; got {prefill_text!r}"
        )
    if generation_prompt_suffix != DEFAULT_GENERATION_PROMPT_SUFFIX:
        raise ValueError(
            "generation_prompt_suffix must be exactly "
            f"{DEFAULT_GENERATION_PROMPT_SUFFIX!r}; got {generation_prompt_suffix!r}. "
            "An empty or altered value would disable the answer-slot check"
        )
    if tuple(control_tokens) != DEFAULT_SCAFFOLD_CONTROL_TOKENS:
        raise ValueError(
            "scaffold_control_tokens must be exactly "
            f"{list(DEFAULT_SCAFFOLD_CONTROL_TOKENS)} in that order; got "
            f"{list(control_tokens)}. An empty or reordered list would disable the "
            "answer-slot check"
        )


def _check_answer_slot_scaffold(
    tokenizer: Any,
    prefill_text: str,
    prefill_ids: torch.Tensor,
    control_tokens: Sequence[str],
    generation_prompt_suffix: str,
) -> None:
    """Verify the tokenizer realizes the production scaffold as production does.

    Mirrors ``score_verifier.ForcedChoiceVerifier.__init__``: the chat template's
    generation prompt must end at ``<|start|>assistant`` and the harmony control
    tokens must survive as single canonical ids, otherwise the appended prefill
    lands somewhere other than ``<|channel|>final<|message|>Answer:`` and GCG would
    silently optimize a letter that the verifier never emits there.

    The scaffold VALUES are pinned separately by ``_require_production_scaffold``, so
    every check below is unconditional -- no argument can switch one off.
    """

    rendered = _chat_text(tokenizer, "probe")
    if rendered is None:
        raise ValueError(
            "tokenizer cannot render the chat template as text, so the "
            f"generation-prompt tail {generation_prompt_suffix!r} cannot be verified"
        )
    if not rendered.endswith(generation_prompt_suffix):
        raise ValueError(
            "chat-template generation prompt must end with "
            f"{generation_prompt_suffix!r} so the final-channel A/B readout is "
            f"positioned like production; got tail {rendered[-120:]!r}"
        )
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    scaffold = [int(x) for x in prefill_ids.tolist()]
    for position, token in enumerate(control_tokens):
        ids = _encode(tokenizer, token)
        if ids.numel() != 1:
            raise ValueError(
                f"harmony control token {token!r} does not tokenize to a single id "
                f"(ids={ids.tolist()}); the prefill scaffold would be mis-tokenized"
            )
        canonical = int(ids[0])
        if callable(convert):
            mapped = convert(token)
            if mapped is None or int(mapped) != canonical:
                raise ValueError(
                    f"harmony control token {token!r} does not map to its canonical id "
                    f"(encode={canonical}, convert_tokens_to_ids={mapped})"
                )
        if canonical not in scaffold:
            raise ValueError(
                f"final-channel prefill {prefill_text!r} does not contain control token "
                f"{token!r} as a single id (ids={scaffold}); refusing to score at a "
                "mis-placed answer slot"
            )
        if position == 0 and scaffold[0] != canonical:
            raise ValueError(
                f"final-channel prefill must start with {token!r} (ids={scaffold}); "
                "refusing to score at a mis-placed answer slot"
            )


def compile_task_spec(
    spec: Mapping[str, Any],
    tokenizer: Any,
    *,
    prefill: str | None = None,
    completions: Mapping[str, str] | None = None,
    edit_marker: str | None = None,
) -> tuple[list[CompiledTask], dict[str, int], str]:
    """Tokenize raw task prompts and locate the editable block.

    The marker is inserted *before* chat templating and located by CHARACTERS in the
    text rendering, then each side is encoded separately. Searching for the marker's
    token run instead would be fragile: a pre-tokenizer that merges the marker's
    leading ``"\\n"`` with the preceding punctuation (``".\\n"``, ``"!\\n"``) destroys
    the run even though the splice point itself is perfectly well defined.

    The character split is only a hypothesis about where the token boundary lands;
    :class:`SuffixValidator` proves it per suffix by re-rendering
    ``prompt_prefix + suffix_text + prompt_suffix`` and requiring token-for-token
    equality with the spliced sequence. Because production tokenizes through
    ``apply_chat_template(tokenize=True)``, compilation additionally requires the
    text and tokenized template paths to agree.
    """

    meta = spec.get("meta", {})
    if not isinstance(meta, Mapping):
        raise ValueError("task spec meta must be an object")
    prefill_text = prefill if prefill is not None else str(meta.get("prefill", DEFAULT_PREFILL))
    completion_map = dict(completions or meta.get("choice_completions", DEFAULT_COMPLETIONS))
    if set(completion_map) != {"A", "B"}:
        raise ValueError("choice completions must contain exactly A and B")
    marker = edit_marker or str(meta.get("edit_marker", DEFAULT_EDIT_MARKER))
    if not marker:
        raise ValueError("editable marker cannot be empty")
    control_tokens = meta.get("scaffold_control_tokens", DEFAULT_SCAFFOLD_CONTROL_TOKENS)
    if isinstance(control_tokens, str) or not isinstance(control_tokens, (list, tuple)):
        raise ValueError("meta.scaffold_control_tokens must be a list of strings")
    control_tokens = [str(x) for x in control_tokens]
    generation_prompt_suffix = meta.get(
        "generation_prompt_suffix", DEFAULT_GENERATION_PROMPT_SUFFIX
    )
    # Pin the whole scaffold to the production contract before touching the tokenizer:
    # neither a caller override nor task meta may point the readout somewhere else.
    _require_production_scaffold(prefill_text, control_tokens, generation_prompt_suffix)

    choice_ids = {
        letter: _token_id(tokenizer, completion_map[letter], letter)
        for letter in ("A", "B")
    }
    if choice_ids["A"] == choice_ids["B"]:
        raise ValueError(f"A/B completions map to the same token id {choice_ids['A']}")
    prefill_ids = _encode(tokenizer, prefill_text)
    if prefill_ids.numel() == 0:
        raise ValueError("final-channel prefill tokenization is empty")
    _check_answer_slot_scaffold(
        tokenizer, prefill_text, prefill_ids, control_tokens, generation_prompt_suffix,
    )
    compiled: list[CompiledTask] = []
    for raw in parsed_tasks(spec):
        if marker in raw.prompt_prefix or marker in raw.prompt_suffix:
            raise ValueError(f"{raw.task_id}: prompt already contains the editable marker")
        content = raw.prompt_prefix + marker + raw.prompt_suffix
        rendered = _chat_text(tokenizer, content)
        if rendered is None:
            raise ValueError(
                f"{raw.task_id}: tokenizer cannot render the chat template as text, so "
                "the editable span cannot be located by characters"
            )
        occurrences = rendered.count(marker)
        if occurrences != 1:
            raise ValueError(
                f"{raw.task_id}: editable marker occurs {occurrences} times in the "
                "rendered chat template (expected exactly 1); refusing an ambiguous "
                "editable span"
            )
        if not torch.equal(_encode(tokenizer, rendered), _chat_ids(tokenizer, content)):
            raise ValueError(
                f"{raw.task_id}: tokenizing the rendered chat template disagrees with "
                "apply_chat_template(tokenize=True); the character-anchored split "
                "would not describe the prompt production actually scores"
            )
        start = rendered.index(marker)
        prefix_text = rendered[:start]
        tail_text = rendered[start + len(marker):]
        if not prefix_text or not tail_text:
            raise ValueError(f"{raw.task_id}: editable span leaves an empty prompt side")
        prefix_ids = _encode(tokenizer, prefix_text).contiguous()
        tail_ids = torch.cat([_encode(tokenizer, tail_text), prefill_ids], dim=0).contiguous()
        if prefix_ids.numel() == 0:
            raise ValueError(f"{raw.task_id}: editable span leaves an empty prompt side")
        compiled.append(
            CompiledTask(
                task_id=raw.task_id,
                group=raw.group,
                prompt_prefix=raw.prompt_prefix,
                prompt_suffix=raw.prompt_suffix,
                prefix_ids=prefix_ids,
                tail_ids=tail_ids,
                prefill_ids=prefill_ids.clone(),
                target_index=0 if raw.target == "A" else 1,
                target_letter=raw.target,
                target_token_id=choice_ids[raw.target],
                weight=raw.weight,
                metadata=raw.metadata,
            )
        )
    return compiled, choice_ids, prefill_text


def assemble_ids(task: CompiledTask, suffix_ids: torch.Tensor) -> torch.Tensor:
    suffix_ids = suffix_ids.detach().to(dtype=torch.long, device="cpu")
    if suffix_ids.ndim != 1 or suffix_ids.numel() == 0:
        raise ValueError("GCG suffix must be a non-empty one-dimensional token tensor")
    return torch.cat([task.prefix_ids, suffix_ids, task.tail_ids], dim=0)


# --------------------------------------------------------------------------- #
# suffix text/token fidelity
# --------------------------------------------------------------------------- #


def _boundary_probe(
    tokenizer: Any, ids: torch.Tensor, side: str, window: int
) -> tuple[str, torch.Tensor] | None:
    """Largest self-stable decode/re-encode window at a splice boundary."""

    total = int(ids.numel())
    for size in range(min(window, total), 0, -1):
        chunk = (ids[-size:] if side == "left" else ids[:size]).contiguous()
        text = str(tokenizer.decode(chunk.tolist(), skip_special_tokens=False))
        if not text or REPLACEMENT_CHAR in text:
            continue
        if torch.equal(_encode(tokenizer, text), chunk):
            return text, chunk
    return None


class SuffixValidator:
    """Fail-closed decode/re-tokenize filter for GCG suffixes.

    ``is_admissible`` is the cheap tier-1 filter (this is the analogue of the
    reference implementation's ``filter_cand``): it rejects any candidate whose text
    form would not reproduce its own token ids, is unprintable, contains transcript
    control substrings, or breaks a splice boundary.  ``verify_exact`` is the tier-2
    ground truth: the full prompt is re-rendered through the chat template and must
    match the spliced ids token for token.
    """

    def __init__(
        self,
        tokenizer: Any,
        tasks: Sequence[CompiledTask],
        *,
        forbidden_substrings: Sequence[str] = FORBIDDEN_SUFFIX_SUBSTRINGS,
        require_leading_whitespace: bool = True,
        probe_window: int = 4,
        edit_marker: str | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("SuffixValidator requires at least one compiled task")
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.forbidden = tuple(forbidden_substrings)
        if edit_marker:
            self.forbidden = self.forbidden + (edit_marker,)
        self.require_leading_whitespace = bool(require_leading_whitespace)
        self.probes_enabled = True
        self.probe_window = int(probe_window)
        probes: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        empty = torch.zeros(0, dtype=torch.long)
        for task in self.tasks:
            left = _boundary_probe(tokenizer, task.prefix_ids, "left", self.probe_window)
            right = _boundary_probe(tokenizer, task.tail_ids, "right", self.probe_window)
            left_text, left_ids = left if left is not None else ("", empty)
            right_text, right_ids = right if right is not None else ("", empty)
            if not left_text and not right_text:
                continue
            probes[(left_text, right_text)] = (left_ids, right_ids)
        self.probes = [
            (left_text, left_ids, right_text, right_ids)
            for (left_text, right_text), (left_ids, right_ids) in probes.items()
        ]

    # -- text form ---------------------------------------------------------- #
    def decode(self, suffix_ids: torch.Tensor) -> str:
        return str(
            self.tokenizer.decode(
                [int(x) for x in suffix_ids.tolist()], skip_special_tokens=False
            )
        )

    def text_reason(self, suffix_ids: torch.Tensor, text: str | None = None) -> str | None:
        """Context-free rejection reason, or None when the text form is sound."""

        if text is None:
            text = self.decode(suffix_ids)
        if not text:
            return "decodes to the empty string"
        if REPLACEMENT_CHAR in text:
            return "decodes to an undecodable byte fragment"
        if self.require_leading_whitespace and not text[:1].isspace():
            return "does not start with whitespace (would glue onto the preceding word)"
        for bad in self.forbidden:
            if bad and bad in text:
                return f"contains the forbidden substring {bad!r}"
        re_encoded = _encode(self.tokenizer, text)
        if not torch.equal(re_encoded, suffix_ids.to(dtype=torch.long, device="cpu")):
            return "does not re-tokenize to its own ids"
        return None

    def probe_reason(self, suffix_ids: torch.Tensor, text: str | None = None) -> str | None:
        if not self.probes_enabled or not self.probes:
            return None
        if text is None:
            text = self.decode(suffix_ids)
        ids = suffix_ids.to(dtype=torch.long, device="cpu")
        for left_text, left_ids, right_text, right_ids in self.probes:
            expected = torch.cat([left_ids, ids, right_ids], dim=0)
            actual = _encode(self.tokenizer, left_text + text + right_text)
            if not torch.equal(actual, expected):
                return "changes tokenization at a splice boundary"
        return None

    def admission_reason(self, suffix_ids: torch.Tensor) -> str | None:
        text = self.decode(suffix_ids)
        return self.text_reason(suffix_ids, text) or self.probe_reason(suffix_ids, text)

    def is_admissible(self, suffix_ids: torch.Tensor) -> bool:
        return self.admission_reason(suffix_ids) is None

    # -- exact, in-context verification ------------------------------------- #
    def exact_reason(self, suffix_ids: torch.Tensor, text: str | None = None) -> str | None:
        """Re-render every task prompt with the suffix as TEXT and compare ids."""

        if text is None:
            text = self.decode(suffix_ids)
        ids = suffix_ids.to(dtype=torch.long, device="cpu")
        for task in self.tasks:
            expected = assemble_ids(task, ids)
            rendered = _chat_ids(
                self.tokenizer, task.prompt_prefix + text + task.prompt_suffix
            )
            actual = torch.cat([rendered, task.prefill_ids], dim=0)
            if not torch.equal(actual, expected):
                return (
                    f"{task.task_id}: re-tokenizing the deployed text yields "
                    f"{int(actual.numel())} tokens that differ from the optimized "
                    f"{int(expected.numel())}-token sequence"
                )
        return None

    def verify_exact(self, suffix_ids: torch.Tensor) -> None:
        reason = self.exact_reason(suffix_ids)
        if reason is not None:
            raise ValueError(f"GCG suffix is not deployable as text: {reason}")

    # -- calibration -------------------------------------------------------- #
    def calibrate(self, suffix_ids: torch.Tensor) -> dict[str, Any]:
        """Check the cheap filter against the exact one on a known suffix.

        If the boundary probes reject a suffix that the exact check accepts, the
        probes are unusable for this tokenizer (e.g. a tokenizer that re-normalizes
        leading whitespace) and are disabled rather than starving the search. The
        exact check still guards everything that is adopted.
        """

        text = self.decode(suffix_ids)
        exact = self.exact_reason(suffix_ids, text)
        text_only = self.text_reason(suffix_ids, text)
        probe = self.probe_reason(suffix_ids, text)
        if exact is None and probe is not None:
            self.probes_enabled = False
        return {
            "probes": len(self.probes),
            "probes_enabled": self.probes_enabled,
            "initial_text_reason": text_only,
            "initial_probe_reason": probe,
            "initial_exact_reason": exact,
        }


def _left_pad(sequences: Sequence[torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence list")
    max_len = max(int(seq.numel()) for seq in sequences)
    if max_len < 1:
        raise ValueError("cannot pad empty input sequences")
    input_ids = torch.full((len(sequences), max_len), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        length = int(seq.numel())
        input_ids[row, max_len - length :] = seq
        attention[row, max_len - length :] = 1
    return input_ids, attention


def _last_logits(logits: torch.Tensor, batch_width: int) -> torch.Tensor:
    if logits.ndim != 3:
        raise RuntimeError(f"model logits must be rank-3, got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return logits[:, 0, :]
    if logits.shape[1] == batch_width:
        # Inputs are left-padded, so every real sequence ends at the final column.
        return logits[:, -1, :]
    raise RuntimeError(
        f"model returned {logits.shape[1]} time positions for a padded width {batch_width}; "
        "cannot identify the final-channel answer slot"
    )


def _call_model(model: Any, *, input_ids: torch.Tensor | None,
                inputs_embeds: torch.Tensor | None,
                attention_mask: torch.Tensor, logits_kwarg: str | None) -> Any:
    # GCG never reuses a KV cache: every forward is a fresh full-sequence pass. Letting
    # the model build one would allocate gigabytes per batch on the 20B run for nothing.
    kwargs: dict[str, Any] = {"attention_mask": attention_mask, "use_cache": False}
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
    if logits_kwarg:
        kwargs[logits_kwarg] = 1
    return model(**kwargs)


def _choice_nll(last_logits: torch.Tensor, choice_ids: Mapping[str, int],
                target_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.stack(
        [last_logits[:, int(choice_ids["A"])], last_logits[:, int(choice_ids["B"])]]
    , dim=-1).float()
    log_p = z - torch.logsumexp(z, dim=-1, keepdim=True)
    loss = -log_p.gather(1, target_index.view(-1, 1)).squeeze(1)
    return loss, log_p


def _group_means(tasks: Sequence[CompiledTask], values: Sequence[float]) -> dict[str, float]:
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for task, value in zip(tasks, values):
        sums[task.group] = sums.get(task.group, 0.0) + task.weight * float(value)
        weights[task.group] = weights.get(task.group, 0.0) + task.weight
    return {group: sums[group] / weights[group] for group in sums}


def _result_from_values(tasks: Sequence[CompiledTask], losses: Sequence[float],
                        target_probs: Sequence[float]) -> ObjectiveResult:
    denominator = sum(task.weight for task in tasks)
    total = sum(task.weight * float(loss) for task, loss in zip(tasks, losses)) / denominator
    return ObjectiveResult(
        loss=float(total),
        per_task_loss=[float(x) for x in losses],
        per_task_target_probability=[float(x) for x in target_probs],
        group_loss=_group_means(tasks, losses),
        group_target_probability=_group_means(tasks, target_probs),
    )


def score_suffixes(
    model: Any,
    tasks: Sequence[CompiledTask],
    suffixes: Sequence[torch.Tensor],
    choice_ids: Mapping[str, int],
    *,
    pad_token_id: int,
    eval_batch_size: int = 8,
    logits_kwarg: str | None = None,
) -> list[ObjectiveResult]:
    """Exact forward score for one or more suffixes.

    ``eval_batch_size`` bounds the number of FORWARD ROWS per model call, not the
    number of candidates: the (candidate, task) cross product is flattened first and
    then chunked, so a run with many tasks cannot silently multiply the batch by the
    task count. Left padding makes the final answer slot the last real token for
    every row, which permits the production-compatible ``logits_to_keep=1``
    optimization when the model exposes it.
    """

    if not tasks:
        raise ValueError("GCG requires at least one task")
    if not suffixes:
        return []
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    rows: list[tuple[int, int]] = [
        (candidate_index, task_index)
        for candidate_index in range(len(suffixes))
        for task_index in range(len(tasks))
    ]
    loss_rows = [[0.0 for _ in tasks] for _ in suffixes]
    prob_rows = [[0.0 for _ in tasks] for _ in suffixes]
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    device = model_device(model)
    try:
        with torch.no_grad():
            for offset in range(0, len(rows), eval_batch_size):
                chunk = rows[offset : offset + eval_batch_size]
                sequences = [
                    assemble_ids(tasks[task_index], suffixes[candidate_index])
                    for candidate_index, task_index in chunk
                ]
                input_ids, attention = _left_pad(sequences, pad_token_id)
                input_ids = input_ids.to(device)
                attention = attention.to(device)
                outputs = _call_model(
                    model, input_ids=input_ids, inputs_embeds=None,
                    attention_mask=attention, logits_kwarg=logits_kwarg,
                )
                last = _last_logits(outputs.logits, input_ids.shape[1])
                target = torch.tensor(
                    [tasks[task_index].target_index for _, task_index in chunk],
                    dtype=torch.long, device=last.device,
                )
                losses, log_p = _choice_nll(last, choice_ids, target)
                probabilities = log_p.gather(1, target.view(-1, 1)).squeeze(1).exp()
                for row, (candidate_index, task_index) in enumerate(chunk):
                    loss_rows[candidate_index][task_index] = float(losses[row].cpu())
                    prob_rows[candidate_index][task_index] = float(probabilities[row].cpu())
    finally:
        if model_was_training:
            model.train()
    return [
        _result_from_values(tasks, row_losses, row_probs)
        for row_losses, row_probs in zip(loss_rows, prob_rows)
    ]


def score_suffix(
    model: Any,
    tasks: Sequence[CompiledTask],
    suffix_ids: torch.Tensor,
    choice_ids: Mapping[str, int],
    *,
    pad_token_id: int,
    eval_batch_size: int = 8,
    logits_kwarg: str | None = None,
) -> ObjectiveResult:
    return score_suffixes(
        model, tasks, [suffix_ids], choice_ids,
        pad_token_id=pad_token_id, eval_batch_size=eval_batch_size,
        logits_kwarg=logits_kwarg,
    )[0]


def model_device(model: Any) -> torch.device:
    """Resolve the device for ordinary and device-mapped HF models."""

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(model, "device", None)
        if device is None:
            return torch.device("cpu")
        return torch.device(device)


def _freeze_model(model: Any) -> dict[int, bool]:
    states: dict[int, bool] = {}
    for parameter in model.parameters():
        states[id(parameter)] = bool(parameter.requires_grad)
        parameter.requires_grad_(False)
    return states


def _restore_model_grad_states(model: Any, states: Mapping[int, bool]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(bool(states.get(id(parameter), False)))


def suffix_gradient(
    model: Any,
    tasks: Sequence[CompiledTask],
    suffix_ids: torch.Tensor,
    choice_ids: Mapping[str, int],
    *,
    logits_kwarg: str | None = None,
    normalize_per_task: bool = True,
) -> torch.Tensor:
    """Return the aggregated gradient with shape ``[suffix_len, hidden_size]``.

    Each task gradient is normalized to unit Frobenius norm before its task weight is
    applied. This is the aggregation used when one suffix is optimized against
    several prompts/readouts; it prevents a long or high-logit task from dominating
    the update merely because of scale. (The reference universal-GCG implementation
    normalizes per token position instead; both are proposal heuristics, and the
    exact rerank downstream is what actually selects the replacement.)
    """

    if not tasks:
        raise ValueError("GCG requires at least one task")
    suffix_ids = suffix_ids.detach().to(dtype=torch.long, device="cpu")
    if suffix_ids.ndim != 1 or suffix_ids.numel() == 0:
        raise ValueError("suffix_ids must be a non-empty one-dimensional tensor")
    embedding = model.get_input_embeddings()
    device = model_device(model)
    aggregate: torch.Tensor | None = None
    total_weight = 0.0
    was_training = bool(getattr(model, "training", False))
    model.eval()
    states = _freeze_model(model)
    try:
        for task in tasks:
            ids = assemble_ids(task, suffix_ids).to(device)
            attention = torch.ones((1, ids.numel()), dtype=torch.long, device=device)
            # Detaching here is important: autograd.grad should expose only the
            # embedding input, never accumulate gradients into the frozen model.
            embeds = embedding(ids.unsqueeze(0)).detach().requires_grad_(True)
            outputs = _call_model(
                model, input_ids=None, inputs_embeds=embeds,
                attention_mask=attention, logits_kwarg=logits_kwarg,
            )
            last = _last_logits(outputs.logits, ids.numel())
            target = torch.tensor([task.target_index], dtype=torch.long, device=device)
            losses, _ = _choice_nll(last, choice_ids, target)
            grad = torch.autograd.grad(losses[0], embeds, retain_graph=False)[0][0]
            start = task.prefix_length
            end = start + int(suffix_ids.numel())
            suffix_grad = grad[start:end]
            if suffix_grad.shape[0] != int(suffix_ids.numel()):
                raise RuntimeError(
                    f"{task.task_id}: gradient suffix span has shape {tuple(suffix_grad.shape)}"
                )
            if normalize_per_task:
                norm = suffix_grad.float().norm()
                if float(norm) > 0.0 and math.isfinite(float(norm)):
                    suffix_grad = suffix_grad / norm.to(suffix_grad.dtype)
            weighted = float(task.weight) * suffix_grad
            aggregate = weighted if aggregate is None else aggregate + weighted
            total_weight += float(task.weight)
    finally:
        _restore_model_grad_states(model, states)
        if was_training:
            model.train()
    if aggregate is None or total_weight <= 0.0:
        raise RuntimeError("GCG gradient aggregation produced no task gradients")
    return aggregate / total_weight


def tokenizer_vocab_size(tokenizer: Any) -> int | None:
    """Largest id the tokenizer can actually encode/decode, if it reports one."""

    try:
        size = len(tokenizer)
        if isinstance(size, int) and size > 0:
            return size
    except TypeError:
        pass
    size = getattr(tokenizer, "vocab_size", None)
    if isinstance(size, int) and size > 0:
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        try:
            highest = max((int(key) for key in added), default=size - 1)
        except (TypeError, ValueError):
            highest = size - 1
        return max(size, highest + 1)
    return None


def _forbidden_special_ids(tokenizer: Any) -> set[int]:
    forbidden = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [] if x is not None)
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            forbidden.update(int(x) for x in value if x is not None)
        else:
            forbidden.add(int(value))
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    try:
        for key, token in added.items():
            if bool(getattr(token, "special", False)):
                forbidden.add(int(key))
    except (AttributeError, TypeError, ValueError):
        pass
    return forbidden


def _token_text_is_safe(text: str, *, allow_non_ascii: bool) -> bool:
    if not text or REPLACEMENT_CHAR in text:
        return False
    if not allow_non_ascii and not text.isascii():
        return False
    for char in text:
        code = ord(char)
        if char in "\n\t":
            continue
        if code < 0x20 or code == 0x7F:
            return False
    for bad in FORBIDDEN_SUFFIX_SUBSTRINGS:
        if bad in text:
            return False
    return True


def allowed_token_ids(
    tokenizer: Any,
    vocab_size: int,
    *,
    allow_special_tokens: bool = False,
    allow_non_ascii: bool = False,
    strict_roundtrip: bool = True,
) -> torch.Tensor:
    """Build the editable vocabulary.

    Excluded by default: ids the tokenizer cannot address (padded embedding rows),
    special/control/padding ids, tokens that do not decode to printable text, tokens
    that do not survive a single-token decode/re-encode round trip, and tokens whose
    text would corrupt the transcript rendering. Everything here is a necessary
    condition for faithful text deployment; the suffix-level filter in
    :class:`SuffixValidator` handles the multi-token interactions.
    """

    if vocab_size < 1:
        raise ValueError("model vocabulary is empty")
    limit = vocab_size
    reported = tokenizer_vocab_size(tokenizer)
    if reported is not None:
        limit = min(limit, int(reported))
    if limit < 1:
        raise ValueError("tokenizer vocabulary is empty")
    if allow_special_tokens and not strict_roundtrip:
        return torch.arange(limit, dtype=torch.long)
    forbidden = set() if allow_special_tokens else _forbidden_special_ids(tokenizer)
    values = [index for index in range(limit) if index not in forbidden]
    if strict_roundtrip and values:
        texts = _decode_batch(tokenizer, [[index] for index in values])
        keep = [
            (index, text)
            for index, text in zip(values, texts)
            if _token_text_is_safe(text, allow_non_ascii=allow_non_ascii)
        ]
        encoded = _encode_batch(tokenizer, [text for _, text in keep])
        values = [
            index
            for (index, _), ids in zip(keep, encoded)
            if len(ids) == 1 and int(ids[0]) == index
        ]
    if not values:
        raise ValueError(
            "every vocabulary token was excluded; relax --allow-special-tokens / "
            "--allow-non-ascii or disable the round-trip filter"
        )
    return torch.tensor(values, dtype=torch.long)


def infer_logits_kwarg(model: Any) -> str | None:
    """Return the supported last-logits kwarg, matching the production resolver."""

    target = getattr(model, "forward", model)
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind in (
            parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY
        ):
            return name
    return None
