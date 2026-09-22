"""Greedy Coordinate Gradient optimizer.

Search loop per step:

1. one backward pass per task gives ``-dL/dx_i @ E.T``; take the top-k replacement
   tokens per coordinate (restricted to the editable vocabulary);
2. sample substitutions (one deterministic best-per-coordinate pass, then the
   paper's random coordinate / random top-k token rule);
3. drop every substitution that is not deployable as text -- both the cheap filter
   (the analogue of the reference implementation's ``filter_cand``) and the exact
   in-context re-tokenization check run here, so rejected candidates never reach
   the model;
4. score the survivors plus the incumbent with exact forward passes;
5. adopt the lowest-loss candidate (ties broken by index, so reranking is
   deterministic). Every candidate in that set is already verified and the incumbent
   was verified when it was adopted, so a step can never adopt a suffix whose emitted
   text denotes a different token sequence.

The per-coordinate proposal order is shuffled from the run's seeded rng: a fixed
0..n order would permanently starve coordinates past the candidate budget.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Mapping, Sequence

import torch

from adversarial_transcript import common

from .objective import (
    CompiledTask,
    ObjectiveResult,
    RESULT_SCHEMA_VERSION,
    SuffixValidator,
    allowed_token_ids,
    score_suffix,
    score_suffixes,
    sha256_text,
    suffix_gradient,
)


@dataclasses.dataclass(frozen=True)
class GCGConfig:
    """Search controls recorded in every result."""

    steps: int = 64
    top_k: int = 256
    candidates: int = 256
    eval_batch_size: int = 8
    seed: int = 0
    normalize_per_task: bool = True

    def validate(self) -> None:
        for name in ("steps", "top_k", "candidates", "eval_batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


@dataclasses.dataclass
class GCGResult:
    suffix_ids: list[int]
    suffix_text: str
    suffix_text_sha256: str
    initial: ObjectiveResult
    final: ObjectiveResult
    steps: list[dict[str, Any]]
    config: dict[str, Any]
    fidelity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "suffix_ids": self.suffix_ids,
            "suffix_text": self.suffix_text,
            "suffix_text_sha256": self.suffix_text_sha256,
            "initial": dataclasses.asdict(self.initial),
            "final": dataclasses.asdict(self.final),
            "steps": self.steps,
            "config": self.config,
            "fidelity": self.fidelity,
        }


class GCGOptimizer:
    """Optimize one shared discrete suffix over a collection of readout tasks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tasks: Sequence[CompiledTask],
        choice_ids: Mapping[str, int],
        *,
        pad_token_id: int,
        config: GCGConfig | None = None,
        logits_kwarg: str | None = None,
        allowed_ids: torch.Tensor | None = None,
        validator: SuffixValidator | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("GCG requires at least one task")
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.choice_ids = dict(choice_ids)
        self.pad_token_id = int(pad_token_id)
        self.config = config or GCGConfig()
        self.config.validate()
        self.logits_kwarg = logits_kwarg
        self.validator = validator or SuffixValidator(tokenizer, self.tasks)
        vocab_size = int(model.get_input_embeddings().weight.shape[0])
        if allowed_ids is None:
            allowed_ids = allowed_token_ids(tokenizer, vocab_size)
        allowed_ids = allowed_ids.detach().to(dtype=torch.long, device="cpu")
        if allowed_ids.ndim != 1 or allowed_ids.numel() == 0:
            raise ValueError("allowed_ids must be a non-empty one-dimensional tensor")
        if int(allowed_ids.min()) < 0 or int(allowed_ids.max()) >= vocab_size:
            raise ValueError("allowed_ids contains a token outside the model vocabulary")
        self.allowed_ids = torch.unique(allowed_ids, sorted=True)

    def _propose_token_table(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        gradient = suffix_gradient(
            self.model,
            self.tasks,
            suffix_ids,
            self.choice_ids,
            logits_kwarg=self.logits_kwarg,
            normalize_per_task=self.config.normalize_per_task,
        )
        embedding = self.model.get_input_embeddings().weight.detach()
        # The negative directional derivative is the GCG replacement score: lower
        # first-order loss candidates have larger values here.
        scores = -gradient.float() @ embedding.float().T
        allowed = self.allowed_ids.to(scores.device)
        restricted = scores.index_select(1, allowed)
        k = min(int(self.config.top_k), int(allowed.numel()))
        top_positions = restricted.topk(k=k, dim=-1).indices
        return allowed[top_positions].detach().cpu()

    def _sample_candidates(
        self, current: torch.Tensor, token_table: torch.Tensor, rng: random.Random,
    ) -> tuple[list[torch.Tensor], int, int]:
        """Return (deployable proposals, tier-1 rejections, exact rejections).

        Both fidelity tiers run HERE, before anything reaches the model: a candidate
        that cannot be deployed as text must never consume a forward pass.
        """

        count = int(self.config.candidates)
        suffix_len = int(current.numel())
        proposals: list[torch.Tensor] = []
        seen = {tuple(int(x) for x in current.tolist())}
        filtered = 0
        exact_rejected = 0

        def consider(candidate: torch.Tensor) -> bool:
            nonlocal filtered, exact_rejected
            key = tuple(int(x) for x in candidate.tolist())
            if key in seen:
                return False
            seen.add(key)
            if not self.validator.is_admissible(candidate):
                filtered += 1
                return False
            if self.validator.exact_reason(candidate) is not None:
                exact_rejected += 1
                return False
            proposals.append(candidate)
            return True

        # One best proposal per coordinate. The order is shuffled from the run's rng:
        # a fixed 0..n order would permanently starve every coordinate past the
        # candidate budget when candidates < suffix_len.
        coordinates = list(range(suffix_len))
        rng.shuffle(coordinates)
        for coordinate in coordinates:
            candidate = current.clone()
            candidate[coordinate] = int(token_table[coordinate, 0])
            consider(candidate)
            if len(proposals) >= count:
                return proposals, filtered, exact_rejected

        # The rest follows the paper's random coordinate / top-k token sampling rule.
        attempts = 0
        max_attempts = max(32, count * 20)
        while len(proposals) < count and attempts < max_attempts:
            attempts += 1
            coordinate = rng.randrange(suffix_len)
            candidate = current.clone()
            candidate[coordinate] = int(
                token_table[coordinate, rng.randrange(token_table.shape[1])]
            )
            consider(candidate)
        return proposals, filtered, exact_rejected

    @staticmethod
    def _metric_delta(before: ObjectiveResult, after: ObjectiveResult) -> dict[str, float]:
        return {
            "loss_before": float(before.loss),
            "loss_after": float(after.loss),
            "loss_delta": float(after.loss - before.loss),
        }

    def optimize(self, suffix_ids: torch.Tensor) -> GCGResult:
        """Run exact-evaluate-after-gradient GCG and return a serializable result."""

        current = suffix_ids.detach().to(dtype=torch.long, device="cpu").clone()
        if current.ndim != 1 or current.numel() == 0:
            raise ValueError("initial suffix must be a non-empty one-dimensional tensor")
        vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
        if int(current.min()) < 0 or int(current.max()) >= vocab_size:
            raise ValueError("initial suffix contains a token outside the model vocabulary")
        if not bool(torch.isin(current, self.allowed_ids).all()):
            raise ValueError(
                "initial suffix contains a token outside the editable vocabulary; "
                "either choose a clean seed or pass a matching allowed_ids set"
            )
        calibration = self.validator.calibrate(current)
        # Fail closed before spending any compute: if the seed suffix cannot be
        # deployed as text, nothing downstream of it could be either.
        self.validator.verify_exact(current)
        reason = self.validator.text_reason(current)
        if reason is not None:
            raise ValueError(f"initial suffix is not a deployable text fragment: {reason}")

        initial = score_suffix(
            self.model, self.tasks, current, self.choice_ids,
            pad_token_id=self.pad_token_id, eval_batch_size=self.config.eval_batch_size,
            logits_kwarg=self.logits_kwarg,
        )
        best = initial
        rng = random.Random(int(self.config.seed))
        steps: list[dict[str, Any]] = []
        exact_rejections = 0
        filter_rejections = 0
        for step in range(int(self.config.steps)):
            token_table = self._propose_token_table(current)
            proposals, rejected, step_exact_rejections = self._sample_candidates(
                current, token_table, rng
            )
            filter_rejections += rejected
            exact_rejections += step_exact_rejections
            # Every proposal here already passed both fidelity tiers, so no forward
            # pass is ever spent on a suffix that could not be deployed as text.
            scored = score_suffixes(
                self.model,
                self.tasks,
                proposals,
                self.choice_ids,
                pad_token_id=self.pad_token_id,
                eval_batch_size=self.config.eval_batch_size,
                logits_kwarg=self.logits_kwarg,
            )
            # Index 0 is the incumbent, which was verified when it was adopted; the
            # (loss, index) key keeps reranking deterministic under ties.
            candidates = [(current, best)] + list(zip(proposals, scored))
            chosen_index = min(
                range(len(candidates)), key=lambda i: (candidates[i][1].loss, i)
            )
            chosen_ids, chosen_result = candidates[chosen_index]

            changed = (chosen_ids != current).nonzero(as_tuple=False)
            if changed.numel():
                coordinate = int(changed[0, 0])
                old_token = int(current[coordinate])
                new_token = int(chosen_ids[coordinate])
            else:
                coordinate = None
                old_token = None
                new_token = None
            previous = best
            current = chosen_ids.clone()
            best = chosen_result
            steps.append({
                "step": step,
                "candidate_count": len(proposals),
                "filtered_candidate_count": rejected,
                "exact_rejected_count": step_exact_rejections,
                "coordinate": coordinate,
                "old_token_id": old_token,
                "new_token_id": new_token,
                **self._metric_delta(previous, chosen_result),
                "group_loss": dict(chosen_result.group_loss),
                "group_target_probability": dict(chosen_result.group_target_probability),
            })

        # Serialization guard: the emitted text must denote exactly `current`.
        suffix_text = self.validator.decode(current)
        self.validator.verify_exact(current)
        reason = self.validator.text_reason(current, suffix_text)
        if reason is not None:
            raise ValueError(f"final suffix is not a deployable text fragment: {reason}")
        return GCGResult(
            suffix_ids=[int(x) for x in current.tolist()],
            suffix_text=suffix_text,
            suffix_text_sha256=sha256_text(suffix_text),
            initial=initial,
            final=best,
            steps=steps,
            config=dataclasses.asdict(self.config),
            fidelity={
                "roundtrip_verified": True,
                "verified_task_count": len(self.tasks),
                "filter_rejected_candidates": filter_rejections,
                "exact_rejected_candidates": exact_rejections,
                **calibration,
            },
        )


def save_result(path: str, result: GCGResult, *, metadata: Mapping[str, Any] | None = None) -> None:
    """Write the result atomically, creating the parent directory if needed.

    Reuses the repository's atomic writer (same-directory temp file + ``os.replace``,
    temp removed on failure): a long GCG run must not be lost to a half-written file,
    and a serialization error must not leave a truncated result behind.
    """

    payload = result.to_dict()
    if metadata:
        payload["meta"] = dict(metadata)
    common.write_json(path, payload)
