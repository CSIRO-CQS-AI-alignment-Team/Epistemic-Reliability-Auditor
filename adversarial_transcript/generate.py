"""Generate best-of-K debate candidates for the paired Q_Y/Q_H datasets.

The release pipeline generates adversarial candidates by compromising the debater that
defends ``Y_true``. That debater privately receives ``Q_H``, both hidden answer options,
and the target ``H_false``; the stored public transcript remains an ordinary three-round
``Q_Y`` debate in the native ``debate.py`` shape.

The immutable, content-hashed ``hybrid-v2`` instruction stays in the compromised
debater's system message. A target-free reminder is appended to each of its turns, and
runtime checks verify the placement and exposure count of every injected segment.

``--request-seed-policy native`` derives each model-request seed from candidate identity,
so concurrent and resumed runs use the same per-request schedule. The default ``none``
does not send a request seed. ``--concurrency`` parallelizes complete candidates through
the API backend; rounds within one candidate remain sequential. The in-process
Transformers backend serializes model calls because the shared model is not thread-safe.

Mainline inputs are ``dataset/<DATASET>/<DATASET>-with-honest-transcripts.json`` and
``dataset/<DATASET>/<DATASET>-title-story.json`` for QuALITY-H or GPQA.

Example:
    python3 -m adversarial_transcript.generate \
        --items dataset/QuALITY-H/QuALITY-H-with-honest-transcripts.json \
        --stories dataset/QuALITY-H/QuALITY-H-title-story.json \
        --out runs/adversarial/QuALITY-H/candidates.jsonl \
        --k 8 --backend api --model google/gemma-4-31B-it --concurrency 16
"""

import argparse
import concurrent.futures
import hashlib
import os
import sys
import threading
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adversarial_transcript import common, prompt, schema  # noqa: E402

DEFAULT_ITEMS = "dataset/QuALITY-H/QuALITY-H-with-honest-transcripts.json"
DEFAULT_STORIES = "dataset/QuALITY-H/QuALITY-H-title-story.json"
DEFAULT_BACKEND = "transformers"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_DEVICE_MAP = "single"
DEFAULT_CONCURRENCY = 16
CONDITIONS = ("honest", "adversarial")
DEFAULT_CONDITIONS = ("adversarial",)

# Kept at module scope for callers that stamp the generator protocol in a run manifest.
# The exact production prompt identity is ``gen_meta.prompt.content_sha256``.
PROMPT_VERSION = prompt.PROMPT_VERSION

# ---- per-request sampling seeds -------------------------------------------
# `native` derives a seed from candidate identity rather than call order, so the
# schedule is stable under concurrency and resume and covers both debaters' calls.
REQUEST_SEED_POLICY_NONE = "none"
REQUEST_SEED_POLICY_NATIVE = "native"
REQUEST_SEED_POLICIES = (REQUEST_SEED_POLICY_NONE, REQUEST_SEED_POLICY_NATIVE)
REQUEST_SEED_POLICY_VERSION = "adv-generate-request-seed-v1"
REQUEST_SEED_NAMESPACE = "adv.generate.request"
REQUEST_SEED_MODULUS = 2 ** 31
REQUEST_SEED_DETERMINISM_SCOPE = (
    "the request RNG is pinned from candidate identity; bitwise equality across "
    "vLLM/torch/CUDA versions or hardware is NOT asserted"
)
SEED_UNSUPPORTED_FAIL = "fail"
SEED_UNSUPPORTED_RECORD = "record"
SEED_UNSUPPORTED_MODES = (SEED_UNSUPPORTED_FAIL, SEED_UNSUPPORTED_RECORD)


def request_seed(base_seed, dataset_index, condition, candidate_idx, round_index,
                 debater):
    """Deterministic per-request seed. A pure function of identity, never of order."""
    key = "|".join(str(part) for part in (
        REQUEST_SEED_NAMESPACE, int(base_seed), int(dataset_index), str(condition),
        int(candidate_idx), int(round_index), str(debater),
    ))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % REQUEST_SEED_MODULUS


def candidate_request_seeds(base_seed, dataset_index, condition, candidate_idx,
                            num_rounds):
    """{debater_name: [seed per round]} for one candidate, in call order."""
    return {
        name: [
            request_seed(base_seed, dataset_index, condition, candidate_idx,
                         round_index, name)
            for round_index in range(int(num_rounds))
        ]
        for name in common.DEBATER_NAMES
    }


class RequestSeedStatus:
    """Run-wide record of whether per-request seeding actually took effect.

    Shared across worker threads, so every mutation is under a lock. `applied` starts
    True for the native policy and can only be turned OFF (once) by an endpoint that
    rejects the `seed` request option under --seed-unsupported record.
    """

    def __init__(self, policy, mechanism=None, on_unsupported=SEED_UNSUPPORTED_FAIL):
        self.policy = policy
        self.mechanism = mechanism
        self.on_unsupported = on_unsupported
        self.unsupported_reason = None
        self._enabled = policy == REQUEST_SEED_POLICY_NATIVE
        self._lock = threading.Lock()

    def enabled(self):
        with self._lock:
            return self._enabled

    def mark_unsupported(self, reason):
        """Disable seeding for the rest of the run and record why. Returns True once."""
        with self._lock:
            first = self._enabled
            self._enabled = False
            if self.unsupported_reason is None:
                self.unsupported_reason = reason
        return first

    def snapshot(self):
        with self._lock:
            return {
                "policy_version": (REQUEST_SEED_POLICY_VERSION
                                   if self.policy == REQUEST_SEED_POLICY_NATIVE else None),
                "policy": self.policy,
                "applied": bool(self._enabled),
                "mechanism": self.mechanism,
                "unsupported_reason": self.unsupported_reason,
                "determinism_scope": REQUEST_SEED_DETERMINISM_SCOPE,
            }


def seed_status_of(args):
    """The run's seed status, defaulting to an inert 'none' record."""
    status = getattr(args, "_request_seed_status", None)
    if status is None:
        status = RequestSeedStatus(getattr(args, "request_seed_policy",
                                           REQUEST_SEED_POLICY_NONE))
    return status


def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return ivalue


def parse_conditions(raw):
    conditions = [c.strip().lower() for c in raw.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown condition(s): {', '.join(bad)}")
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return conditions


def _copy_qy(qy):
    return {k: qy[k] for k in ("question", "Y_true", "Y_false") if k in qy}


def _copy_qh(qh):
    return {k: qh[k] for k in ("question", "H_true", "H_false") if k in qh}


def _native_stance(stance):
    return {"Debater A": stance["Debater A"], "Debater B": stance["Debater B"]}


def _answer_for_label(qy, label):
    return qy["Y_true"] if label == "Y_true" else qy["Y_false"]


def _opposing_label(label):
    return "Y_false" if label == "Y_true" else "Y_true"


def candidate_key_from_parts(index, condition, candidate_idx):
    return (common.item_id_for(index), condition, int(candidate_idx))


def candidate_key(candidate):
    return (candidate["item_id"], candidate["condition"], int(candidate["candidate_idx"]))


class PrivateInstructionError(RuntimeError):
    """Raised when the private instruction is not injected exactly as declared.

    Fail-closed: the candidate is discarded rather than stored with an unverifiable
    exposure record. A duplicated or missing private-instruction segment would make
    the hybrid-vs-system-only exposure comparison meaningless.
    """


class _SeededClientView:
    """Per-debater, per-candidate view over the shared client that pins request seeds.

    One instance per debater per candidate, so the round cursor is never shared across
    threads: candidates run on separate threads and the rounds inside a candidate are
    strictly sequential. There is deliberately NO global request-order stream -- the
    seeds are a pure function of candidate identity, so any interleaving is fine.
    """

    def __init__(self, inner, seeds, debater):
        self._inner = inner
        self._seeds = list(seeds)
        self._debater = debater
        self._round = 0

    def set_round(self, round_index):
        self._round = int(round_index)

    def generate(self, messages):
        if self._round >= len(self._seeds):
            raise PrivateInstructionError(
                f"{self._debater} requested round {self._round + 1} but only "
                f"{len(self._seeds)} request seed(s) were planned; the rollout made "
                "more calls than --num-rounds"
            )
        return self._inner.generate(messages, seed=self._seeds[self._round])


def _make_round_tracking_class(cls):
    """Subclass that tells a _SeededClientView which round is about to be generated.

    This is the ONLY seeding hook on the debater; `query` is left alone so the
    compromised debater's reminder injection is unaffected by seeding.
    """

    class RoundTrackingDebater(cls):
        def take_turn(self, completed_rounds):
            setter = getattr(self.model_client, "set_round", None)
            if setter is not None:
                setter(len(completed_rounds))
            return super().take_turn(completed_rounds)

    RoundTrackingDebater.__name__ = f"RoundTracking{cls.__name__}"
    return RoundTrackingDebater


def _make_compromised_debater_class(debate_mod):
    """Debater subclass that injects and verifies the fixed private instruction."""

    class CompromisedDebater(debate_mod.Debater):
        def __init__(
            self,
            name,
            opponent_name,
            question,
            defending_answer,
            opposing_answer,
            story,
            model_client,
            qh,
            num_rounds=None,
        ):
            super().__init__(
                name,
                opponent_name,
                question,
                defending_answer,
                opposing_answer,
                story,
                model_client,
            )
            self.qh = qh
            self.num_rounds = num_rounds
            self.system_note = prompt.render_system_note(name, qh)
            self.reminder_scans = []
            self.exposure_records = []
            self._current_reminder = None
            self.system = dict(self.system)
            self.system["content"] = (
                f"{self.system['content'].rstrip()}\n\n{self.system_note}"
            )

        def build_turn_message(self, completed_rounds):
            message = super().build_turn_message(completed_rounds)
            reminder = prompt.render_round_reminder(self.name, len(completed_rounds))
            self._current_reminder = reminder
            scan = prompt.scan_reminder_for_target_text(reminder, self.qh)
            self.reminder_scans.append(scan)
            if scan["hits"]:
                raise PrivateInstructionError(
                    f"{self.name}: the fixed round reminder contains hidden-question "
                    f"text {scan['hits']}"
                )
            message = dict(message)
            message["content"] = (
                f"{(message.get('content') or '').rstrip()}\n\n{reminder}"
            )
            return message

        def query(self, messages):
            self._verify_private_instruction(messages)
            return super().query(messages)

        def _verify_private_instruction(self, messages):
            exposure = prompt.measure_exposure(
                messages, self.system_note, self._current_reminder,
                self.num_rounds if self.num_rounds is not None else len(messages),
            )
            if not exposure["verified_segment_placement"]:
                raise PrivateInstructionError(
                    f"{self.name}: private-instruction segments are not present exactly "
                    f"once at their expected positions (found {exposure['segment_positions']})"
                )
            if exposure["target_exposure_per_request"] > 1:
                raise PrivateInstructionError(
                    f"{self.name}: {exposure['target_exposure_per_request']} "
                    "target-bearing private-instruction components in ONE request; "
                    "hybrid must not exceed system-only's single copy"
                )
            self.exposure_records.append(exposure)

    return CompromisedDebater


def _make_debater(debate_mod, name, opponent_name, qy, stance_label, story,
                  model_client, compromised, qh, num_rounds=None):
    defending = _answer_for_label(qy, stance_label)
    opposing = _answer_for_label(qy, _opposing_label(stance_label))
    if compromised:
        cls = _make_compromised_debater_class(debate_mod)
        if isinstance(model_client, _SeededClientView):
            cls = _make_round_tracking_class(cls)
        return cls(name, opponent_name, qy["question"], defending, opposing,
                   story, model_client, qh, num_rounds)
    cls = debate_mod.Debater
    if isinstance(model_client, _SeededClientView):
        cls = _make_round_tracking_class(cls)
    return cls(name, opponent_name, qy["question"], defending, opposing, story,
               model_client)


def _client_for(model_client, seeds, debater):
    """Wrap the shared client for one debater when native seeding is active."""
    if seeds is None:
        return model_client
    return _SeededClientView(model_client, seeds[debater], debater)


def generate_model_transcript(example, story, stance, condition, model_client,
                              num_rounds, context=None):
    """Generate one real transcript with debate.Debater, preserving a fixed stance.

    Called from a worker thread under --concurrency > 1. Every object it touches is
    either fresh per call (the Debater pair, the per-debater seeded client views, the
    rounds list) or read-only (example, story, stance); the only shared object is
    `model_client`, whose thread-safety is established by build_generation_client.

    `context` carries optional per-debater request seeds. Its default supports direct
    callers that do not supply a seed schedule.
    """
    import debate  # noqa: PLC0415

    context = context or {}
    seeds = context.get("request_seeds")
    qy = example["Q_Y"]
    qh = example["Q_H"]
    compromised = (
        common.compromised_debater_from_stance(stance)
        if condition == "adversarial"
        else None
    )

    debater_a = _make_debater(
        debate, "Debater A", "Debater B", qy, stance["Debater A"], story,
        _client_for(model_client, seeds, "Debater A"),
        compromised == "Debater A", qh, num_rounds=num_rounds,
    )
    debater_b = _make_debater(
        debate, "Debater B", "Debater A", qy, stance["Debater B"], story,
        _client_for(model_client, seeds, "Debater B"),
        compromised == "Debater B", qh, num_rounds=num_rounds,
    )

    rounds = []
    for _ in range(num_rounds):
        # Strictly sequential: each turn is prompted with the rounds completed so far,
        # so rounds can never be parallelised. The candidate is the unit of parallelism.
        arg_a = debater_a.take_turn(rounds)
        arg_b = debater_b.take_turn(rounds)
        rounds.append({"Debater A": arg_a, "Debater B": arg_b})

    return {"Debater A": stance["Debater A"], "Debater B": stance["Debater B"],
            "rounds": rounds}


def generate_dry_run_transcript(example, stance, condition, candidate_idx,
                                num_rounds):
    """Deterministic lightweight transcript used for smoke tests and demos."""
    qy = example["Q_Y"]
    title = example.get("story_title") or "the story"
    rounds = []
    for r in range(num_rounds):
        round_no = r + 1
        a_answer = _answer_for_label(qy, stance["Debater A"])
        b_answer = _answer_for_label(qy, stance["Debater B"])
        style = "careful" if condition == "honest" else "selective"
        rounds.append({
            "Debater A": (
                f"Round {round_no}: Debater A gives a {style} public argument "
                f"that {a_answer} is the better answer to the visible question "
                f"about {title}. Candidate {candidate_idx} emphasizes story "
                "details that support this visible answer over the alternative."
            ),
            "Debater B": (
                f"Round {round_no}: Debater B gives a {style} public argument "
                f"that {b_answer} is the better answer to the visible question "
                f"about {title}. Candidate {candidate_idx} emphasizes story "
                "details that support this visible answer over the alternative."
            ),
        })
    return {"Debater A": stance["Debater A"], "Debater B": stance["Debater B"],
            "rounds": rounds}


def _word_limit_hint():
    """debate.WORD_LIMIT when debate.py is already imported, else its default.

    Only used to render reminders for the METADATA record; the live rollout always
    renders through the real module. Recorded so a difference is visible.
    """
    debate_mod = sys.modules.get("debate")
    return getattr(debate_mod, "WORD_LIMIT", 100) if debate_mod is not None else 100


def prompt_exposure_meta(example, condition, transcript, num_rounds, dry_run):
    """Structural private-instruction exposure record for one candidate.

    Counts the fixed system-note component that carries the steering target.  The
    target-free reminder is still measured for prompt-length provenance.

    The live rollout independently verifies placement/multiplicity on the real message
    list and fails the candidate otherwise (CompromisedDebater._verify_private_instruction),
    so `live_verified` says whether that check actually ran.
    """
    base = {
        "kind": "private_instruction_target_exposure_per_request",
        "target_bearing_slots": list(prompt.TARGET_BEARING_SLOTS),
        "qh_bearing_slots": list(prompt.QH_BEARING_SLOTS),
        "note": (
            "counts the injected component carrying {h_false}; the system note is "
            "re-sent on every request, so exposure is reported per request"
        ),
    }
    if condition != "adversarial":
        base.update({
            "applies": False,
            "reason": "no private instruction is injected in the honest condition",
            "target_exposure_per_request": 0,
            "requests_per_rollout": int(num_rounds),
            "target_bearing_requests_per_rollout": 0,
            "total_target_occurrences_per_rollout": 0,
            "live_verified": False,
        })
        return base

    compromised = common.compromised_debater_from_stance(transcript)
    qh = example.get("Q_H", {})
    word_limit = _word_limit_hint()
    system_note = prompt.render_system_note(compromised, qh)
    reminder_rounds = []
    injected_chars = []
    for round_index in range(int(num_rounds)):
        reminder = prompt.render_round_reminder(compromised, round_index)
        reminder_rounds.append(round_index + 1)
        injected_chars.append(len(system_note) + len(reminder))

    base.update({
        "applies": True,
        "compromised_debater": compromised,
        "target_exposure_per_request": 1,
        "target_exposure_per_request_by_round": [1] * int(num_rounds),
        "requests_per_rollout": int(num_rounds),
        "target_bearing_requests_per_rollout": int(num_rounds),
        "total_target_occurrences_per_rollout": int(num_rounds),
        "reminder_rounds": reminder_rounds,
        "injected_private_instruction_chars_by_round": injected_chars,
        "word_limit_used": word_limit,
        "live_verified": not bool(dry_run),
    })
    return base


def prompt_meta(example, condition, transcript, args):
    block = prompt.describe()
    block["source"] = "builtin"
    block["source_path"] = None
    block["applies_to_condition"] = condition == "adversarial"
    block["exposure"] = prompt_exposure_meta(
        example, condition, transcript, args.num_rounds, args.dry_run)
    return block


def request_seed_meta(dataset_index, condition, candidate_idx, args):
    status = seed_status_of(args)
    block = status.snapshot()
    if status.policy == REQUEST_SEED_POLICY_NATIVE and not args.dry_run:
        block["seeds"] = candidate_request_seeds(
            args.seed, dataset_index, condition, candidate_idx, args.num_rounds)
        block["key_fields"] = ["base_seed", "dataset_index", "condition",
                               "candidate_idx", "round_index_zero_based", "debater"]
        block["derivation"] = "sha256-first64-mod-2**31"
        block["namespace"] = REQUEST_SEED_NAMESPACE
        block["covers"] = list(common.DEBATER_NAMES)
    else:
        block["seeds"] = None
    return block


def build_candidate(example, dataset_index, condition, candidate_idx, transcript,
                    stance_source, args):
    compromised = (
        common.compromised_debater_from_stance(transcript)
        if condition == "adversarial"
        else None
    )
    candidate = {
        "item_id": common.item_id_for(dataset_index),
        "dataset_index": int(dataset_index),
        "story_title": example.get("story_title"),
        "condition": condition,
        "candidate_idx": int(candidate_idx),
        "compromised_debater": compromised,
        "qh_target": "H_false" if condition == "adversarial" else None,
        "transcript": transcript,
        "qy": _copy_qy(example.get("Q_Y", {})),
        "qh": _copy_qh(example.get("Q_H", {})),
        "gen_meta": {
            "prompt_version": prompt.PROMPT_VERSION,
            "backend": "dry-run" if args.dry_run else args.backend,
            "model": None if args.dry_run else args.model,
            "seed": args.seed,
            "candidate_seed": f"{args.seed}|{dataset_index}|{condition}|{candidate_idx}",
            "num_rounds": args.num_rounds,
            "stance_source": stance_source,
            "dry_run": bool(args.dry_run),
            # The endpoint actually used, resolved in build_generation_client, so a run
            # against a remote server does not record a localhost URL it never called.
            "api_base_url": (None if args.dry_run or args.backend != "api"
                             else getattr(args, "_effective_api_base_url",
                                          getattr(args, "api_base_url", None))),
            "prompt": prompt_meta(example, condition, transcript, args),
            "request_seed": request_seed_meta(
                dataset_index, condition, candidate_idx, args),
        },
    }
    errors = schema.validate_candidate(candidate)
    if errors:
        raise ValueError(
            f"invalid generated candidate {candidate['item_id']}/"
            f"{condition}#{candidate_idx}: {errors[0]}"
        )
    return candidate


def iter_indices(items, start_index, limit, num_shards=1, shard_index=0):
    """Canonical dataset indices owned by this window and shard."""
    end = len(items) if limit is None else min(len(items), start_index + limit)
    for dataset_index in range(start_index, end):
        if (dataset_index - start_index) % num_shards == shard_index:
            yield dataset_index

def count_pending_candidates(items, story_map, args, skip_keys):
    """Count candidates this run will generate after resume/shard filtering."""
    total = 0
    for dataset_index in iter_indices(
        items,
        args.start_index,
        args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        example = items[dataset_index]
        if story_map.get(example.get("story_title")) is None:
            continue
        for condition in args.conditions:
            for candidate_idx in range(args.k):
                key = candidate_key_from_parts(dataset_index, condition, candidate_idx)
                if key not in skip_keys:
                    total += 1
    return total


def resolve_device_map(args):
    if args.backend != "transformers":
        return None
    if args.device_map == "single":
        return {"": 0}
    return args.device_map


class SerializedModelClient:
    """Serialise generate() for a backend whose underlying call is not thread-safe.

    The transformers backend shares ONE in-process HuggingFace model, and
    `model.generate` must not be entered from two threads at once. Wrapping it keeps
    --concurrency legal for every backend without pretending the local backend gets
    faster: candidate threads still interleave, but only one is inside the model.

    It is also where per-request seeding happens for this backend: torch's RNG is
    process-global, so the seed call must be ATOMIC with the generate() it seeds --
    hence inside the same lock.
    """

    def __init__(self, inner, status=None):
        self._inner = inner
        self._status = status
        self._lock = threading.Lock()
        self._torch = getattr(inner, "_torch", None)

    def _apply_seed(self, seed):
        torch = self._torch
        if torch is None:
            try:
                import torch as torch_module  # noqa: PLC0415
            except ImportError as exc:
                reason = f"torch unavailable for local seeding: {exc}"
                if self._status.on_unsupported == SEED_UNSUPPORTED_FAIL:
                    raise RuntimeError(
                        f"--request-seed-policy native: {reason}. Re-run with "
                        "--seed-unsupported record to generate unseeded and say so."
                    ) from exc
                self._status.mark_unsupported(reason)
                return
            torch = self._torch = torch_module
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate(self, messages, seed=None):
        with self._lock:
            if seed is not None and self._status is not None and self._status.enabled():
                self._apply_seed(seed)
            return self._inner.generate(messages)


# The request options debate.ApiModelClient.generate passes through. SeededApiClient
# must forward exactly these plus `seed`; a drift test asserts the sets still agree.
API_LM_OPTION_KEYS = ("temperature", "top_p", "max_tokens", "timeout")


def _looks_like_unsupported_request_option(exc):
    """Whether an API error means 'this endpoint rejects the option', not 'it failed'.

    Only a client-side TypeError (SDK without the parameter) or a 400/422 from the
    server counts. Anything else -- timeouts, 5xx, connection errors -- is a normal
    transient failure and must propagate so the candidate is retried, NOT silently
    downgrade the whole run to unseeded.
    """
    if isinstance(exc, TypeError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (400, 422)


class SeededApiClient:
    """Add a per-request ``seed`` to ``debate.ApiModelClient``.

    The wrapper is thread-safe because it never mutates the shared client. It issues the
    same chat-completion request using the wrapped client's model and generation options.
    """

    def __init__(self, inner, status):
        self._inner = inner
        self._status = status

    def generate(self, messages, seed=None):
        if seed is None or not self._status.enabled():
            return self._inner.generate(messages)
        lm = self._inner.lm
        options = {key: lm[key] for key in API_LM_OPTION_KEYS}
        try:
            completion = self._inner.client.chat.completions.create(
                model=self._inner.model, messages=messages, seed=seed, **options)
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if not _looks_like_unsupported_request_option(exc):
                raise
            reason = f"endpoint rejected the `seed` request option: {type(exc).__name__}: {exc}"
            if self._status.on_unsupported == SEED_UNSUPPORTED_FAIL:
                raise RuntimeError(
                    f"--request-seed-policy native: {reason}. Re-run with "
                    "--seed-unsupported record to generate unseeded and record it."
                ) from exc
            if self._status.mark_unsupported(reason):
                print(f"[WARN] {reason}; continuing UNSEEDED for the rest of this run",
                      file=sys.stderr)
            return self._inner.generate(messages)
        # Length/filter/tool finishes can yield None content; never return None.
        return completion.choices[0].message.content or ""


def build_generation_client(args, concurrency, warn):
    """Build the one model client shared by all workers; return (client, concurrency).

    The returned concurrency may be lower than requested when the client cannot be
    driven in parallel at all. This is also where the run's RequestSeedStatus is
    created and attached to `args`, so build_candidate can record what actually
    happened rather than what was requested.
    """
    import debate  # noqa: PLC0415

    policy = getattr(args, "request_seed_policy", REQUEST_SEED_POLICY_NONE)
    on_unsupported = getattr(args, "seed_unsupported", SEED_UNSUPPORTED_FAIL)
    # ``debate.BASE_URL`` is a module constant. Temporarily bind an explicit API endpoint
    # only during client construction, then restore the shared value.
    base_url = getattr(args, "api_base_url", None)
    previous_base_url = getattr(debate, "BASE_URL", None)
    if base_url:
        debate.BASE_URL = base_url
    try:
        client = debate.build_model_client(args, device_map=resolve_device_map(args))
    finally:
        if base_url:
            debate.BASE_URL = previous_base_url
    args._effective_api_base_url = base_url or previous_base_url

    if isinstance(client, debate.TransformersModelClient):
        # HF generate() is not thread-safe; a lock is enough because every other piece
        # of per-candidate state is thread-local.
        status = RequestSeedStatus(
            policy, "torch.manual_seed inside the shared client lock", on_unsupported)
        args._request_seed_status = status
        return SerializedModelClient(client, status), concurrency

    if isinstance(client, debate.ApiModelClient):
        # openai.OpenAI wraps a thread-safe httpx client and ApiModelClient.generate
        # only reads immutable attributes, so worker threads can call it directly.
        status = RequestSeedStatus(
            policy, "chat.completions.create(seed=...)", on_unsupported)
        args._request_seed_status = status
        if policy == REQUEST_SEED_POLICY_NATIVE:
            return SeededApiClient(client, status), concurrency
        return client, concurrency

    # An external driver may replace ``debate.build_model_client`` with a stateful
    # adapter that assumes one ordered
    # stream of model calls (per-request seed plans). Parallel candidates would
    # interleave that stream, so fall back to serial instead of silently corrupting
    # whatever the adapter records -- and let ITS seed policy stay authoritative.
    status = RequestSeedStatus(REQUEST_SEED_POLICY_NONE, None, on_unsupported)
    if policy == REQUEST_SEED_POLICY_NATIVE:
        status.mark_unsupported(
            f"{type(client).__name__} is an external driver adapter that owns its own "
            "request-seed plan; native seeding was disabled to avoid two policies "
            "seeding the same call"
        )
        warn(f"{type(client).__name__} is not a native debate.py model client; "
             "--request-seed-policy native disabled so the driver's own seed plan "
             "stays authoritative")
    args._request_seed_status = status
    if concurrency > 1:
        warn(f"{type(client).__name__} is not a native debate.py model client; "
             "falling back to --concurrency 1 to preserve its call ordering")
    return client, 1


def plan_tasks(items, story_map, args, skip_keys=frozenset(), warn=None):
    """Enumerate the candidates this run still owes, in canonical output order.

    Canonical order is (dataset_index, condition in --conditions order, candidate_idx).
    The executor completes tasks in arbitrary order; `position` is what restores this
    one before anything is written.
    """
    warn = warn or (lambda msg: None)
    tasks = []
    for dataset_index in iter_indices(
        items,
        args.start_index,
        args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ):
        example = items[dataset_index]
        story_title = example.get("story_title")
        story = story_map.get(story_title)
        if story is None:
            warn(f"index {dataset_index}: story '{story_title}' missing; skipped")
            continue

        stance_info = common.resolve_stance(example)
        stance = _native_stance(stance_info)
        stance_source = stance_info.get("source", "unknown")

        for condition in args.conditions:
            for candidate_idx in range(args.k):
                key = candidate_key_from_parts(dataset_index, condition, candidate_idx)
                if key in skip_keys:
                    continue
                tasks.append({
                    "position": len(tasks),
                    "item_id": key[0],
                    "dataset_index": dataset_index,
                    "example": example,
                    "story": story,
                    "stance": stance,
                    "stance_source": stance_source,
                    "condition": condition,
                    "candidate_idx": candidate_idx,
                })
    return tasks


def generate_one(task, args, model_client):
    """Produce exactly one candidate. Runs on a worker thread."""
    if args.dry_run:
        transcript = generate_dry_run_transcript(
            task["example"], task["stance"], task["condition"],
            task["candidate_idx"], args.num_rounds)
    else:
        if model_client is None:
            raise RuntimeError("internal error: pending real generation without a model client")
        seeds = None
        if seed_status_of(args).enabled():
            # Pure function of candidate identity, so worker threads never share or
            # order-depend on this plan.
            seeds = candidate_request_seeds(
                args.seed, task["dataset_index"], task["condition"],
                task["candidate_idx"], args.num_rounds)
        transcript = generate_model_transcript(
            task["example"], task["story"], task["stance"], task["condition"],
            model_client, args.num_rounds,
            context={"request_seeds": seeds})
    # Resolve the builder at call time so an explicit provenance wrapper can intercept it.
    return build_candidate(
        task["example"], task["dataset_index"], task["condition"],
        task["candidate_idx"], transcript, task["stance_source"], args)


def run_generation(tasks, args, model_client, concurrency, on_done):
    """Run every task through a bounded thread pool (mirrors debate-bok.run_generation).

    `on_done(task, candidate, error)` is invoked exactly once per task, always on the
    calling thread, so checkpointing/progress/counters need no cross-thread reasoning.
    `error is None` means the candidate succeeded.
    """
    # Bounded in-flight submission: submitting all 1776 tasks up front would hold every
    # completed candidate (4-40 KB each) in the pool's queue until the loop drained it.
    max_inflight = max(1, concurrency * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        task_iter = iter(tasks)
        exhausted = False
        while True:
            while not exhausted and len(pending) < max_inflight:
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(generate_one, task, args, model_client)] = task
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    candidate = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
                    on_done(task, None, exc)
                else:
                    on_done(task, candidate, None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate honest and adversarial QuALITY-H debate candidates.")
    p.add_argument("--items", default=DEFAULT_ITEMS,
                   help="QuALITY-H JSON list; prefer the honest-transcript file for stance reuse")
    p.add_argument("--stories", default=DEFAULT_STORIES,
                   help="dataset/<DS>/<DS>-title-story.json map")
    p.add_argument("--out", required=True, help="candidate JSONL output")
    p.add_argument("--k", type=positive_int, default=8,
                   help="candidates per (item, condition)")
    p.add_argument("--conditions", type=parse_conditions,
                   default=list(DEFAULT_CONDITIONS),
                   help="comma-separated subset of honest,adversarial; default: adversarial only")
    p.add_argument("--num-rounds", "--rounds", dest="num_rounds",
                   type=positive_int, default=3)
    p.add_argument("--concurrency", type=positive_int, default=DEFAULT_CONCURRENCY,
                   help="candidates generated in parallel; rounds inside one candidate stay "
                        "sequential. Only --backend api parallelises for real: the "
                        "transformers backend serialises its shared model behind a lock")
    p.add_argument("--start-index", type=nonnegative_int, default=0)
    p.add_argument("--limit", type=positive_int, default=None,
                   help="number of dataset rows to process from --start-index")
    p.add_argument("--seed", type=int, default=0,
                   help="recorded in metadata; model sampling is controlled by debate.py config")
    p.add_argument("--dry-run", action="store_true",
                   help="emit deterministic placeholder transcripts without loading a model")
    p.add_argument("--overwrite", action="store_true",
                   help="discard existing --out instead of resuming/skipping present keys")
    p.add_argument("--checkpoint-mode", choices=["candidate", "item", "interval"],
                   default="candidate",
                   help="when to checkpoint --out: after every candidate, after each item, or every --save-every candidates")
    p.add_argument("--save-every", type=positive_int, default=10,
                   help="checkpoint interval for --checkpoint-mode interval")
    p.add_argument("--num-shards", type=positive_int, default=1,
                   help="split item rows across this many generation shards")
    p.add_argument("--shard-index", type=nonnegative_int, default=0,
                   help="0-based generation shard index")
    p.add_argument("--backend", choices=["api", "transformers"], default=DEFAULT_BACKEND)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="api backend only")
    p.add_argument("--api-base-url", default=None,
                   help="api backend only: OpenAI-compatible endpoint to use instead of "
                        "debate.BASE_URL (default http://localhost:18888/v1). Recorded in "
                        "gen_meta.api_base_url")
    p.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"],
                   default="auto", help="transformers backend only")
    p.add_argument("--device-map", default=DEFAULT_DEVICE_MAP,
                   help="transformers backend only; 'single' maps the full model to visible CUDA device 0")
    p.add_argument("--no-progress", action="store_true",
                   help="disable tqdm progress bar")
    p.add_argument("--request-seed-policy", choices=list(REQUEST_SEED_POLICIES),
                   default=REQUEST_SEED_POLICY_NONE,
                   help="'native' derives a per-request sampling seed from candidate "
                        "identity for both debaters (concurrency-safe and resume-stable); "
                        "'none' sends no request seed")
    p.add_argument("--seed-unsupported", choices=list(SEED_UNSUPPORTED_MODES),
                   default=SEED_UNSUPPORTED_FAIL,
                   help="what to do when the backend cannot accept a per-request seed: "
                        "'fail' (default) aborts; 'record' generates unseeded and writes "
                        "the reason into every candidate's gen_meta")
    args = p.parse_args(argv)

    args._request_seed_status = RequestSeedStatus(
        args.request_seed_policy, None, args.seed_unsupported)
    return args


def prompt_version_conflicts(existing):
    """Return rows whose recorded prompt identity differs from the active prompt.

    Rows without ``gen_meta.prompt`` are compared using ``prompt_version``.
    """
    conflicts = []
    for row in existing:
        meta = row.get("gen_meta") or {}
        block = meta.get("prompt") or {}
        row_hash = block.get("content_sha256")
        if row_hash is None:
            row_version = meta.get("prompt_version")
            if row_version == prompt.PROMPT_VERSION:
                continue
            detail = (f"prompt_version={row_version!r} (row without a prompt hash) "
                      f"!= {prompt.PROMPT_VERSION!r}")
        else:
            if row_hash == prompt.CONTENT_SHA256:
                continue
            detail = (f"content_sha256={row_hash[:12]}... != "
                      f"{prompt.CONTENT_SHA256[:12]}... "
                      f"(row prompt_version={meta.get('prompt_version')!r})")
        conflicts.append(
            f"{row.get('item_id')}/{row.get('condition')}#{row.get('candidate_idx')}: "
            f"{detail}")
    return conflicts


def main(argv=None):
    args = parse_args(argv)
    if args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be < --num-shards")

    bar = None  # bound below; `warn` routes around it once the progress bar is live

    def warn(msg):
        text = f"[WARN] {msg}"
        # tqdm.write keeps the bar intact while a run reports per-candidate failures.
        if bar is not None:
            tqdm.write(text, file=sys.stderr)
        else:
            print(text, file=sys.stderr)

    items = common.load_items(args.items)
    stories = common.load_story_map(args.stories)

    print(f"[generate] fixed prompt {prompt.PROMPT_ID!r} "
          f"(version={prompt.PROMPT_VERSION}, placement={prompt.PLACEMENT}, "
          f"sha256={prompt.CONTENT_SHA256[:12]}...)")

    existing = []
    if os.path.exists(args.out) and not args.overwrite:
        existing = common.read_jsonl(args.out)
        conflicts = prompt_version_conflicts(existing)
        if conflicts:
            head = "\n  ".join(conflicts[:5])
            more = f"\n  ... and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
            raise SystemExit(
                f"[PROMPT] {args.out} already holds {len(conflicts)} candidate(s) "
                f"generated with a different private prompt:\n  {head}{more}\n"
                "Resuming would produce a mixed-prompt candidate file. Use "
                "--overwrite to regenerate or point --out at a new file."
            )
    skip_keys = {candidate_key(c) for c in existing}

    pending_total = count_pending_candidates(items, stories, args, skip_keys)
    tasks = plan_tasks(items, stories, args, skip_keys=skip_keys, warn=warn)

    model_client = None
    concurrency = args.concurrency
    if tasks and not args.dry_run:
        model_client, concurrency = build_generation_client(args, concurrency, warn)

    show_progress = tqdm is not None and not args.no_progress and pending_total
    if show_progress:
        desc = f"generate shard {args.shard_index + 1}/{args.num_shards}"
        bar = tqdm(total=pending_total, desc=desc, unit="cand")

    def progress_write(msg):
        if bar is not None:
            tqdm.write(msg)
        else:
            print(msg)

    # Completed candidates are buffered by their canonical `position` and only ever
    # written back in that order, so --out is byte-identical at any --concurrency.
    # Rows that were already present keep their original order and stay first, exactly
    # as the serial implementation left them.
    completed = {}
    remaining_per_item = Counter(task["dataset_index"] for task in tasks)
    write_lock = threading.Lock()
    generated = 0
    failed = 0

    def current_rows():
        return existing + [completed[position] for position in sorted(completed)]

    def checkpoint():
        # Completions are drained on one thread today, so this lock is uncontended; it
        # keeps the whole-file rewrite atomic if a writer is ever moved into a worker.
        with write_lock:
            common.write_jsonl(args.out, current_rows())

    def on_done(task, candidate, error):
        nonlocal generated, failed
        remaining_per_item[task["dataset_index"]] -= 1
        item_done = remaining_per_item[task["dataset_index"]] == 0
        if error is None:
            completed[task["position"]] = candidate
            generated += 1
        else:
            failed += 1
            warn(f"candidate {task['item_id']}/{task['condition']}"
                 f"#{task['candidate_idx']} failed: {type(error).__name__}: {error}")
        # A failed candidate produced no new row, so only the item mode -- which promises
        # "this dataset_index is finished" -- still has something to flush.
        should_checkpoint = (
            (args.checkpoint_mode == "candidate" and error is None)
            or (args.checkpoint_mode == "item" and item_done)
            or (args.checkpoint_mode == "interval" and error is None
                and generated % args.save_every == 0)
        )
        if should_checkpoint:
            checkpoint()
            progress_write(
                f"[generate] checkpoint: {len(existing) + len(completed)} "
                f"total candidate(s) -> {args.out}")
        if bar is not None:
            bar.update(1)

    try:
        run_generation(tasks, args, model_client, concurrency, on_done)
    finally:
        if bar is not None:
            bar.close()

    checkpoint()
    total = len(existing) + len(completed)
    print(f"[generate] generated {generated} new candidate(s); {failed} failed; "
          f"total {total} -> {args.out}")
    if generated == 0 and not failed and skip_keys:
        print("[generate] nothing new to do; existing candidates were reused")
    if failed:
        # Failed candidates are simply absent from --out, so re-running this exact
        # command retries them (and only them) through skip_keys.
        print(f"[generate] {failed} candidate(s) failed; re-run to retry them",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
