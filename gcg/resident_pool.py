"""One-process, multi-GPU resident worker pool for the whole-candidate GCG run.

The enclosing Slurm batch step already owns every requested GPU.  This driver sees
those devices through the batch step's ``CUDA_VISIBLE_DEVICES`` and uses logical
``cuda:0..N-1`` directly; it never shells out to ``srun`` and never creates another
Slurm job step.

Each :class:`ResidentWorker` owns a distinct HF model replica and tokenizer.  The
replica is loaded exactly once, drains GCG candidates from a shared queue, stays
resident while the parent validates/merges the applied candidates, and is then
reused by the canonical forced-choice scorer.  Model replicas are loaded serially:
Hugging Face ``from_pretrained`` uses process-global initialization machinery that is
not safe to run concurrently in multiple threads.  After loading, a
``ThreadPoolExecutor`` provides GCG/scoring concurrency without copying the
already-large Python/HF state into another set of launcher processes.  Models are
never shared between concurrent threads: one worker object, one contiguous GPU group,
one model, and at most one in-flight operation.  With the default four-worker shape
each group contains one GPU.  With ``--workers 2`` the default groups are
``(cuda:0, cuda:1)`` and ``(cuda:2, cuda:3)``; each model is balanced across its
two-card group for both GCG and scoring.

A strict ``[--start, --end)`` manifest subrange runs only the optimization stage.
This lets disjoint cluster nodes share one run root without racing on the global
merged/scored artifacts.  A later unbounded invocation is the explicit finalization
barrier and resumes the verified per-candidate artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import queue
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from tqdm import tqdm

from adversarial_transcript import score_verifier

from . import run as run_module
from . import run_all
from .optimizer import GCGConfig


# Hugging Face/Accelerate model construction temporarily changes process-global
# PyTorch initialization state while materializing meta tensors.  The pipeline
# already calls ``ResidentWorker.load`` serially, but keep this lock at the actual
# loading boundary as a fail-safe for callers that construct replicas concurrently.
_HF_LOAD_LOCK = threading.Lock()
_GPU_MEMORY_RESERVE_BYTES = 4 * 1024**3


def _gpus_per_worker(args: argparse.Namespace) -> int:
    """Return the requested contiguous GPU group size for each worker.

    Two workers form the model-parallel layout for a four-GPU allocation: worker 0
    owns logical GPUs 0 and 1, and worker 1 owns logical GPUs 2 and 3. Other worker
    counts default to one GPU per worker. The explicit option makes the topology
    visible in launch scripts and rejects mismatched allocations.
    """

    configured = getattr(args, "gpus_per_worker", None)
    if configured is None:
        configured = 2 if int(args.workers) == 2 else 1
    configured = int(configured)
    if configured < 1:
        raise SystemExit("--gpus-per-worker must be positive")
    return configured


def _worker_gpu_groups(args: argparse.Namespace) -> tuple[tuple[int, ...], ...]:
    """Return contiguous logical CUDA groups, one group per resident worker."""

    configured = getattr(args, "gpu_groups", None)
    if configured is not None:
        groups = tuple(tuple(int(gpu) for gpu in group) for group in configured)
        if len(groups) != int(args.workers) or any(not group for group in groups):
            raise SystemExit("internal GPU group configuration does not match --workers")
        return groups

    group_size = _gpus_per_worker(args)
    return tuple(
        tuple(range(worker * group_size, (worker + 1) * group_size))
        for worker in range(int(args.workers))
    )


def _expected_gpu_count(args: argparse.Namespace) -> int:
    return sum(len(group) for group in _worker_gpu_groups(args))


def _group_max_memory(gpu_indices: Sequence[int]) -> dict[int, int]:
    """Limit HF's automatic placement to the selected GPUs.

    ``device_map='balanced'`` otherwise considers every visible GPU in the process,
    which would let a worker steal cards from its sibling.  Query free memory at
    load time and leave headroom for activations/gradients; the returned mapping
    contains only the worker's GPUs, so model weights and dispatched forward/backward
    computation stay inside that group.
    """

    max_memory: dict[int, int] = {}
    for gpu_index in gpu_indices:
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(int(gpu_index))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cannot query free memory for logical cuda:{gpu_index} while "
                "building a model-parallel worker"
            ) from exc
        usable = int(free_bytes) - _GPU_MEMORY_RESERVE_BYTES
        if usable <= 0:
            raise RuntimeError(
                f"logical cuda:{gpu_index} has only {int(free_bytes)} free bytes; "
                f"need more than {_GPU_MEMORY_RESERVE_BYTES} bytes of headroom"
            )
        max_memory[int(gpu_index)] = usable
    return max_memory


def _validate_model_parallel_map(model: Any, gpu_indices: Sequence[int]) -> None:
    """Fail closed unless HF dispatched the replica across exactly this group."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, Mapping):
        raise RuntimeError(
            "model-parallel load did not expose an hf_device_map; refusing to run "
            "a --gpus-per-worker > 1 worker without verifying placement"
        )

    used: set[int] = set()
    non_cuda: list[str] = []
    for module_name, destination in device_map.items():
        if isinstance(destination, int):
            used.add(int(destination))
            continue
        if isinstance(destination, torch.device):
            if destination.type == "cuda" and destination.index is not None:
                used.add(int(destination.index))
            else:
                non_cuda.append(f"{module_name}={destination}")
            continue
        rendered = str(destination)
        if rendered.startswith("cuda:"):
            try:
                used.add(int(rendered.split(":", 1)[1]))
            except ValueError:
                non_cuda.append(f"{module_name}={rendered}")
        elif rendered.isdigit():
            used.add(int(rendered))
        else:
            non_cuda.append(f"{module_name}={rendered}")

    expected = {int(index) for index in gpu_indices}
    if non_cuda:
        preview = ", ".join(non_cuda[:4])
        raise RuntimeError(
            "model-parallel placement used CPU/disk/another non-CUDA target: "
            f"{preview}"
        )
    unexpected = sorted(used - expected)
    missing = sorted(expected - used)
    if unexpected or missing:
        raise RuntimeError(
            "model-parallel placement does not match the requested GPU group "
            f"{list(gpu_indices)} (used={sorted(used)}, "
            f"unexpected={unexpected}, missing={missing})"
        )


def _gcg_config(args: argparse.Namespace) -> GCGConfig:
    return GCGConfig(
        steps=args.steps,
        top_k=args.top_k,
        candidates=args.gcg_candidates,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        normalize_per_task=not args.no_gradient_normalization,
    )


def _merge_score_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "merge-scores",
        "--run-root", args.run_root,
        "--stories", args.stories,
        "--model", args.model,
        "--num-shards", str(args.workers),
        "--option-seed", str(args.option_seed),
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


@dataclasses.dataclass
class ResidentWorker:
    """One persistent model replica bound to one contiguous logical CUDA group."""

    gpu_index: int
    device: torch.device
    model: Any
    tokenizer: Any
    args: argparse.Namespace
    progress_callback: Callable[[str, str], None] | None = dataclasses.field(
        default=None, repr=False
    )
    log_callback: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )
    gpu_indices: tuple[int, ...] = dataclasses.field(default_factory=tuple)
    device_map: str | Mapping[str, str] = "single"

    def __post_init__(self) -> None:
        # Default to the declared primary GPU when no explicit group is supplied.
        if not self.gpu_indices:
            self.gpu_indices = (int(self.gpu_index),)
        else:
            self.gpu_indices = tuple(int(index) for index in self.gpu_indices)

    @classmethod
    def load(cls, gpu_index: int, args: argparse.Namespace) -> "ResidentWorker":
        gpu_group = _worker_gpu_groups(args)[int(gpu_index)]
        if args.allow_cpu and not torch.cuda.is_available():
            device = torch.device("cpu")
            device_map: str | Mapping[str, str] = "single"
            max_memory = None
        else:
            device = torch.device(f"cuda:{gpu_group[0]}")
            # ``balanced`` is deliberate: ``auto`` may keep a model entirely on
            # the first card when it fits there, whereas this mode must make both
            # cards in the worker's group participate in the replica.
            device_map = "single" if len(gpu_group) == 1 else "balanced"
            max_memory = (
                None
                if len(gpu_group) == 1
                else _group_max_memory(gpu_group)
            )
        dtype = run_module._dtype(args.dtype)
        if dtype is None and device.type == "cuda":
            dtype = torch.bfloat16
        thread_name = threading.current_thread().name
        print(
            f"[gcg.pool] worker={gpu_index} thread={thread_name} loading "
            f"{args.model} once on {device}"
            + (
                f" across GPUs {list(gpu_group)}"
                if len(gpu_group) > 1
                else ""
            )
        )
        with _HF_LOAD_LOCK:
            tokenizer = run_module._load_tokenizer(args.model)
            # device_map alone is insufficient for MXFP4 -> bf16 materialization: some
            # kernels follow the loading thread's *current* CUDA device.  Pin the whole
            # load context or replica 1+ can accidentally execute materialization on GPU 0.
            load_context = (
                torch.cuda.device(device)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with load_context:
                model = run_module._load_model(
                    args.model,
                    device,
                    dtype,
                    device_map,
                    max_memory=max_memory,
                )
            if len(gpu_group) > 1 and device.type == "cuda":
                _validate_model_parallel_map(model, gpu_group)
        print(
            f"[gcg.pool] worker={gpu_index} model ready on {device}"
            + (
                f" across GPUs {list(gpu_group)}"
                if len(gpu_group) > 1
                else ""
            )
        )
        return cls(
            gpu_index=int(gpu_index),
            device=device,
            model=model,
            tokenizer=tokenizer,
            args=args,
            gpu_indices=gpu_group,
            device_map=device_map,
        )

    def optimize(
        self,
        layout: run_all.RunLayout,
        manifest_rows: Iterable[dict[str, Any]],
    ) -> run_all.ShardOutcome:
        config = _gcg_config(self.args)
        dtype = run_module._dtype(self.args.dtype)
        if dtype is None and self.device.type == "cuda":
            dtype = torch.bfloat16
        execution_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with execution_context:
            log = self.log_callback or (
                lambda message: print(f"[gpu:{self.gpu_index}] {message}")
            )
            return run_all.run_shard(
                layout,
                manifest_rows,
                model=self.model,
                tokenizer=self.tokenizer,
                config=config,
                model_name=self.args.model,
                suffix_len=self.args.suffix_len,
                init_text=self.args.init_text,
                allow_special_tokens=self.args.allow_special_tokens,
                allow_non_ascii=self.args.allow_non_ascii,
                device=self.device,
                device_map=self.device_map,
                dtype=dtype,
                force=self.args.force,
                shard_index=self.gpu_index,
                num_shards=self.args.workers,
                log=log,
                on_candidate_done=self.progress_callback,
            )

    def score(self, layout: run_all.RunLayout) -> int:
        execution_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with execution_context:
            # Release cached allocator blocks after optimization without unloading the
            # resident verifier weights needed for scoring.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            scorer = None
            if not self.args.dry_run:
                primary_gpu = (
                    int(self.device.index)
                    if self.device.type == "cuda" and self.device.index is not None
                    else int(self.gpu_index)
                )
                scorer = score_verifier.ForcedChoiceVerifier(
                    self.args.model,
                    primary_gpu,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
            primary_gpu = (
                int(self.device.index)
                if self.device.type == "cuda" and self.device.index is not None
                else int(self.gpu_index)
            )
            argv = [
                "--candidates", layout.merged_candidates,
                "--stories", self.args.stories,
                "--out", layout.score_shard(self.gpu_index),
                "--model", self.args.model,
                "--conditions", run_all.DEFAULT_CONDITION,
                "--device", str(primary_gpu),
                "--option-seed", str(self.args.option_seed),
                "--num-shards", str(self.args.workers),
                "--shard-index", str(self.gpu_index),
                "--save-every", str(self.args.save_every),
                "--strict-schema",
            ]
            if self.args.dry_run:
                argv.append("--dry-run")
            if self.args.force:
                argv.append("--overwrite")
            print(
                f"[gcg.pool] worker={self.gpu_index} scoring shard "
                f"{self.gpu_index}/{self.args.workers} on GPUs "
                f"{list(self.gpu_indices)} with resident weights"
            )
            return int(score_verifier.main(argv, scorer=scorer) or 0)


def _parallel(
    executor: concurrent.futures.ThreadPoolExecutor,
    calls: Sequence[tuple[str, Callable[[], Any]]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Run every call, wait for all siblings, then report all failures together."""

    futures = {executor.submit(call): label for label, call in calls}
    results: dict[str, Any] = {}
    errors: list[str] = []
    for future in concurrent.futures.as_completed(futures):
        label = futures[future]
        try:
            results[label] = future.result()
        except BaseException as exc:  # keep sibling GPU work durable before failing
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(f"{stage} failed:\n  " + "\n  ".join(sorted(errors)))
    return results


def _validate_devices(args: argparse.Namespace) -> None:
    if args.allow_cpu:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable inside the batch step; the Slurm job must allocate "
            f"and expose {_expected_gpu_count(args)} GPUs"
        )
    expected = _expected_gpu_count(args)
    visible = int(torch.cuda.device_count())
    if visible != expected:
        raise SystemExit(
            f"expected exactly {expected} GPUs in the batch step for "
            f"{args.workers} worker(s) x {_gpus_per_worker(args)} GPU(s), but torch "
            f"sees {visible}; do not call srun here--fix the sbatch GPU "
            "allocation/visibility"
        )


def _drain(work: queue.Queue[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield candidates from the shared queue until another worker drained it."""

    while True:
        try:
            yield work.get_nowait()
        except queue.Empty:
            return


def _seed_then_drain(
    seed: dict[str, Any] | None,
    work: queue.Queue[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Give every available worker one row, then switch to work stealing."""

    if seed is not None:
        yield seed
    yield from _drain(work)


def _gpu_artifacts_already_complete(
    args: argparse.Namespace,
    layout: run_all.RunLayout,
    manifest: Sequence[dict[str, Any]],
) -> bool:
    """Validate a complete resume before paying for any checkpoint load."""

    if not _gcg_artifacts_already_complete(args, layout, manifest):
        return False

    # All GCG artifacts verify against the current model/config. Rebuild the merged
    # candidate view and try the equally strict score merge. A missing/stale score
    # shard means models are still needed; a successful merge proves zero GPU work.
    run_all.main(["merge-candidates", "--run-root", args.run_root])
    try:
        run_all.main(_merge_score_argv(args))
    except SystemExit as exc:
        print(f"[gcg.pool] resume precheck: scoring still needs GPU work ({exc})")
        return False
    print(
        "[gcg.pool] resume precheck: GCG and score artifacts fully verify; "
        "skipping all model loads"
    )
    return True


def _gcg_artifacts_already_complete(
    args: argparse.Namespace,
    layout: run_all.RunLayout,
    manifest: Sequence[dict[str, Any]],
) -> bool:
    """Return whether every candidate in one requested range is reusable."""

    if args.force:
        return False
    config = _gcg_config(args)
    config_payload = run_all.gcg_config_payload(
        config,
        suffix_len=args.suffix_len,
        init_text=args.init_text,
        allow_special_tokens=args.allow_special_tokens,
        allow_non_ascii=args.allow_non_ascii,
    )
    if any(
        run_all.completion_reason(
            layout,
            manifest_row,
            gcg_config=config_payload,
            model_name=args.model,
        )
        is not None
        for manifest_row in manifest
    ):
        return False
    return True


def _manifest_batch(
    args: argparse.Namespace,
    manifest: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Resolve a fail-closed, zero-based ``[start, end)`` manifest slice."""

    total = len(manifest)
    start = int(args.start)
    end = total if args.end is None else int(args.end)
    if start >= total:
        raise SystemExit(
            f"--start must be smaller than the manifest length ({total}), got {start}"
        )
    if end > total:
        raise SystemExit(
            f"--end must not exceed the manifest length ({total}), got {end}"
        )
    if end <= start:
        raise SystemExit(
            f"--end must be greater than --start for the half-open range, "
            f"got [{start}, {end})"
        )
    return list(manifest[start:end]), start, end


def run_pipeline(
    args: argparse.Namespace,
    *,
    worker_loader: Callable[[int, argparse.Namespace], ResidentWorker] | None = None,
) -> int:
    """Optimize one range; finalize merge/scoring only when the scope is full."""

    layout = run_all.RunLayout(args.run_root)
    manifest = run_all.read_manifest(layout)
    batch, start, end = _manifest_batch(args, manifest)
    full_scope = start == 0 and end == len(manifest)
    if full_scope:
        if _gpu_artifacts_already_complete(args, layout, manifest):
            return 0
    elif _gcg_artifacts_already_complete(args, layout, batch):
        print(
            f"[gcg.pool] partial range [{start}, {end}) already verifies; "
            "skipping all model loads"
        )
        return 0
    _validate_devices(args)
    loader = worker_loader or ResidentWorker.load
    print(
        f"[gcg.pool] one batch process, {args.workers} worker threads, "
        f"visible_gpus={torch.cuda.device_count()}, "
        f"gpu_groups={list(_worker_gpu_groups(args))}, "
        f"range=[{start}, {end}), candidates={len(batch)}/{len(manifest)}"
    )

    stage_count = 4 if full_scope else 2
    print(
        f"[gcg.pool] stage 1/{stage_count}: load {args.workers} persistent model replicas "
        "serially"
    )
    load_started = time.monotonic()
    workers: list[ResidentWorker] = []
    for gpu_index in range(args.workers):
        # AutoModel.from_pretrained/Accelerate temporarily patch process-global
        # PyTorch initialization hooks while materializing meta tensors.  Concurrent
        # calls can race and leave a replica partly on the meta device.  Keep only
        # checkpoint construction serial; every expensive GCG/scoring call below is
        # still submitted to the multi-GPU thread pool.
        try:
            workers.append(loader(gpu_index, args))
        except Exception as exc:
            raise RuntimeError(
                "model loading failed:\n  "
                f"gpu-{gpu_index}: {type(exc).__name__}: {exc}"
            ) from exc
    print(f"[gcg.pool] model-load wall time: {time.monotonic() - load_started:.1f}s")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="gcg-gpu",
    ) as executor:
        print(
            f"[gcg.pool] stage 2/{stage_count}: optimize range [{start}, {end}) "
            "from one dynamic queue with "
            f"{args.workers} resident replicas"
        )
        optimize_started = time.monotonic()
        work: queue.Queue[dict[str, Any]] = queue.Queue()
        seeds = list(batch[: args.workers])
        for manifest_row in batch[args.workers :]:
            work.put(manifest_row)
        progress_counts = {"completed": 0, "skipped": 0, "failed": 0}
        progress_lock = threading.Lock()
        with tqdm(
            total=len(batch),
            desc="[gcg.pool] optimize",
            unit="candidate",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            file=sys.stderr,
        ) as progress:
            def on_candidate_done(_stem: str, status: str) -> None:
                with progress_lock:
                    progress_counts[status] += 1
                    progress.update(1)
                    progress.set_postfix(
                        done=progress_counts["completed"],
                        skipped=progress_counts["skipped"],
                        failed=progress_counts["failed"],
                        refresh=False,
                    )

            for worker in workers:
                worker.progress_callback = on_candidate_done
                worker.log_callback = progress.write
            try:
                optimized = _parallel(
                    executor,
                    [
                        (
                            f"gpu-{worker.gpu_index}",
                            lambda worker=worker: worker.optimize(
                                layout,
                                _seed_then_drain(
                                    seeds[worker.gpu_index]
                                    if worker.gpu_index < len(seeds)
                                    else None,
                                    work,
                                ),
                            ),
                        )
                        for worker in workers
                    ],
                    stage="GCG optimization",
                )
            finally:
                for worker in workers:
                    worker.progress_callback = None
                    worker.log_callback = None
        failed = sorted(
            stem
            for outcome in optimized.values()
            for stem in outcome.failed
        )
        if failed:
            preview = ", ".join(failed[:20])
            raise RuntimeError(
                f"GCG optimization failed for {len(failed)} candidate(s): {preview}"
            )
        optimize_elapsed = time.monotonic() - optimize_started
        for label, outcome in sorted(optimized.items()):
            print(
                f"[gcg.pool] {label} optimize: completed={len(outcome.completed)} "
                f"skipped={len(outcome.skipped)} failed={len(outcome.failed)}"
            )
        newly_optimized = sum(len(outcome.completed) for outcome in optimized.values())
        print(
            f"[gcg.pool] optimize wall time: {optimize_elapsed:.1f}s; "
            f"new candidates/s={newly_optimized / max(optimize_elapsed, 1e-9):.6f}"
        )

        if not full_scope:
            print(
                f"[gcg.pool] partial range [{start}, {end}) complete; global "
                "merge/scoring intentionally skipped. After every disjoint range "
                "finishes, rerun once without --start/--end to finalize the pool."
            )
            return 0

        # This CPU validation intentionally runs while the executor and all model
        # objects remain alive; the next submitted calls therefore reuse the weights.
        print("[gcg.pool] stage 3/4: validate and merge applied candidates")
        run_all.main(["merge-candidates", "--run-root", args.run_root])

        print("[gcg.pool] stage 4/4: score with the same resident replicas")
        score_started = time.monotonic()
        scored = _parallel(
            executor,
            [
                (
                    f"gpu-{worker.gpu_index}",
                    lambda worker=worker: worker.score(layout),
                )
                for worker in workers
            ],
            stage="verifier scoring",
        )
        bad_codes = {
            label: int(code) for label, code in scored.items() if int(code) != 0
        }
        if bad_codes:
            raise RuntimeError(f"verifier scoring returned non-zero codes: {bad_codes}")
        score_elapsed = time.monotonic() - score_started
        print(
            f"[gcg.pool] scoring wall time: {score_elapsed:.1f}s "
            "(per-shard logs report new vs resumed rows)"
        )

        run_all.main(_merge_score_argv(args))

    print(
        f"[gcg.pool] GPU stages complete; loaded {args.workers} model replicas total "
        f"(GPU groups={list(_worker_gpu_groups(args))}, reused for both GCG and scoring)"
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gcg.resident_pool",
        description=(
            "Use one Slurm batch process and a resident one-model-per-GPU-group "
            "thread pool for GCG optimization plus verifier scoring."
        ),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stories", required=True)
    parser.add_argument("--model", default=run_all.DEFAULT_MODEL)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument(
        "--gpus-per-worker", type=_positive_int, default=None,
        help=(
            "number of contiguous logical GPUs assigned to each worker; defaults "
            "to 2 when --workers=2 (groups [0,1] and [2,3]), otherwise 1"
        ),
    )
    parser.add_argument(
        "--start", type=_nonnegative_int, default=0,
        help=(
            "zero-based first manifest row to optimize (inclusive); a strict "
            "subrange runs optimization only"
        ),
    )
    parser.add_argument(
        "--end", type=_nonnegative_int, default=None,
        help=(
            "zero-based manifest row at which to stop (exclusive); must not exceed "
            "the manifest length"
        ),
    )
    parser.add_argument("--steps", type=_positive_int, default=run_all.DEFAULT_STEPS)
    parser.add_argument("--top-k", type=_positive_int, default=run_all.DEFAULT_TOP_K)
    parser.add_argument(
        "--gcg-candidates", type=_positive_int,
        default=run_all.DEFAULT_GCG_CANDIDATES,
    )
    parser.add_argument(
        "--eval-batch-size", type=_positive_int,
        default=run_all.DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument("--suffix-len", type=_positive_int, default=run_all.DEFAULT_SUFFIX_LEN)
    parser.add_argument("--seed", type=int, default=run_all.DEFAULT_SEED)
    parser.add_argument("--init-text", default=None)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--allow-non-ascii", action="store_true")
    parser.add_argument("--no-gradient-normalization", action="store_true")
    parser.add_argument(
        "--option-seed", type=int, default=score_verifier.DEFAULT_OPTION_SEED,
    )
    parser.add_argument("--save-every", type=_positive_int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="skip CUDA visibility checks (CPU-stub development only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="use score_verifier's deterministic fake scoring (development only)",
    )
    args = parser.parse_args(argv)
    # Materialize the normalized groups once so launchers/tests can inspect the
    # effective allocation, while helpers still tolerate hand-built Namespaces.
    args.gpus_per_worker = _gpus_per_worker(args)
    args.gpu_groups = _worker_gpu_groups(args)
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    worker_loader: Callable[[int, argparse.Namespace], ResidentWorker] | None = None,
) -> int:
    return run_pipeline(parse_args(argv), worker_loader=worker_loader)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[gcg.pool] FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
