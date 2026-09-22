"""Content-addressed checkpoint fingerprints for verifier readout caches.

A checkpoint path is not a stable identity because training may replace weights in place.
Production fingerprints therefore stream SHA-256 over every weight shard and relevant
configuration, index, tokenizer, and training-metadata file.

Modes:

* ``content`` hashes full weights and is required for production readouts;
* ``metadata`` hashes names, sizes, and small metadata files for diagnostics;
* ``dry-run`` creates an explicit non-production identity without reading a checkpoint.

Fingerprint sidecars are keyed by absolute path and per-file name, size, modification
time, and mode. They are replaced atomically so concurrent jobs can safely cache the same
shared base checkpoint.
"""

import fnmatch
import hashlib
import json
import os

CHUNK = 8 << 20  # 8 MiB

MODE_CONTENT = "content"
MODE_METADATA = "metadata"
MODE_DRY_RUN = "dry-run"
MODES = (MODE_CONTENT, MODE_METADATA, MODE_DRY_RUN)

WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth")
META_PATTERNS = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "pytorch_model.bin.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "training-metadata.json",
)


class FingerprintError(RuntimeError):
    pass


def _matches(name, patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def sha256_file(path, chunk=CHUNK):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def scan(checkpoint_dir):
    """Sorted (relname, size, mtime_ns, ctime_ns, kind) for weight + metadata files.

    `ctime_ns` is in the tuple because mtime alone is forgeable: a same-size overwrite
    followed by `touch -t` restores the mtime and would leave a stale cache entry looking
    valid. Inode change time moves on any write and cannot be reset by utime().
    """
    out = []
    root = os.path.abspath(checkpoint_dir)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if _matches(name, WEIGHT_PATTERNS):
                kind = "weight"
            elif _matches(name, META_PATTERNS):
                kind = "meta"
            else:
                continue
            try:
                stat = os.stat(full)
            except OSError:
                continue
            out.append((rel, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, kind))
    return sorted(out)


def validate_weights(checkpoint_dir, entries):
    """Structural problems that make a directory unusable as a real checkpoint.

    A directory with one matching filename is not a checkpoint. A zero-byte shard hashes
    fine and loads as garbage, and an HF index that names shards which are not present
    describes a torn or partial download. Either would otherwise be stamped
    `production: true` and silently anchor a whole run's cache and resume keys.
    """
    problems = []
    weights = [(rel, size) for rel, size, _m, _c, kind in entries if kind == "weight"]
    if not weights:
        problems.append("no weight files found")
    empty = sorted(rel for rel, size in weights if size == 0)
    if empty:
        problems.append(f"zero-byte weight shard(s): {empty[:5]}")

    present = {rel for rel, _size in weights}
    for index_name in INDEX_FILES:
        index_path = os.path.join(checkpoint_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, ValueError) as exc:
            problems.append(f"{index_name} is unreadable ({type(exc).__name__})")
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            problems.append(f"{index_name} has no weight_map object")
            continue
        referenced = sorted(set(weight_map.values()))
        missing = sorted(shard for shard in referenced if shard not in present)
        if missing:
            problems.append(
                f"{index_name} references {len(missing)} missing shard(s): {missing[:5]}")
    return problems


def _cache_path(cache_dir, checkpoint_dir):
    key = hashlib.sha256(os.path.abspath(checkpoint_dir).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{key}.json")


def shared_cache_dir(out_dir):
    """One fingerprint cache per dataset run tree."""
    return os.path.join(out_dir, "_fingerprints")


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def dry_run_fingerprint(label):
    """An EXPLICIT non-production identity. Never derived from a checkpoint."""
    return {
        "path": label,
        "exists": False,
        "mode": MODE_DRY_RUN,
        "production": False,
        # `identity()` prefixes the mode, so the digest itself carries no prefix.
        "fingerprint": _digest({"label": label})[:32],
        "weights_digest": None,
        "meta_digest": None,
        "n_weight_files": 0,
        "total_weight_bytes": 0,
        "weight_problems": [],
        "hash_source": "dry-run",
        "note": "explicit dry-run identity; carries no checkpoint content",
    }


def fingerprint(checkpoint_dir, mode=MODE_CONTENT, cache_dir=None, force=False,
                allow_missing=False):
    """Fingerprint a checkpoint directory. Reproducible: no timestamps in the result."""
    if mode not in (MODE_CONTENT, MODE_METADATA):
        raise FingerprintError(
            f"fingerprint() takes {MODE_CONTENT!r} or {MODE_METADATA!r}; use "
            f"dry_run_fingerprint() for {MODE_DRY_RUN!r}")
    if not os.path.isdir(checkpoint_dir):
        if not allow_missing:
            raise FingerprintError(
                f"checkpoint directory not found: {checkpoint_dir!r}. Pass "
                "allow_missing=True only for plumbing checks.")
        return {"path": checkpoint_dir, "exists": False, "mode": mode,
                "production": False, "fingerprint": None, "weights_digest": None,
                "meta_digest": None, "n_weight_files": 0, "total_weight_bytes": 0,
                "weight_problems": ["checkpoint directory absent"],
                "hash_source": "missing"}

    entries = scan(checkpoint_dir)
    cache_key = {"path": os.path.abspath(checkpoint_dir), "mode": mode,
                 "entries": [[name, size, mtime, ctime]
                             for name, size, mtime, ctime, _ in entries]}
    cache_file = _cache_path(cache_dir, checkpoint_dir) if cache_dir else None
    if cache_file and not force and os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("cache_key") == cache_key:
                out = dict(cached["fingerprint"])
                out["hash_source"] = "cache"
                return out
        except (OSError, ValueError, KeyError):
            pass  # a corrupt entry is simply recomputed

    problems = validate_weights(checkpoint_dir, entries)
    weights, metas, total_bytes = [], [], 0
    for rel, size, _mtime, _ctime, kind in entries:
        full = os.path.join(checkpoint_dir, rel)
        if kind == "weight":
            total_bytes += size
            weights.append([rel, size, sha256_file(full) if mode == MODE_CONTENT else None])
        else:
            metas.append([rel, size, sha256_file(full)])

    weights_digest = _digest(weights)
    meta_digest = _digest(metas)
    out = {
        "path": checkpoint_dir,
        "exists": True,
        "mode": mode,
        "production": mode == MODE_CONTENT and bool(weights) and not problems,
        "weight_problems": problems,
        "n_weight_files": len(weights),
        "total_weight_bytes": total_bytes,
        "weights_digest": weights_digest,
        "meta_digest": meta_digest,
        "fingerprint": _digest({"mode": mode, "weights": weights_digest,
                                "meta": meta_digest}),
        "hash_source": "computed",
    }
    if cache_file:
        _atomic_write_json(cache_file, {"cache_key": cache_key, "fingerprint": out})
    return out


def identity(fp):
    """The short string used inside cache keys and resume keys."""
    if not fp or not fp.get("fingerprint"):
        raise FingerprintError(
            "refusing to build a cache/resume identity from an empty fingerprint; a real "
            "run must never fall back to a path-only checkpoint id")
    return f"{fp['mode']}:{fp['fingerprint'][:32]}"


def require_production(fp, what="checkpoint", allow_nonproduction=False):
    """Fail closed unless the fingerprint is production-grade."""
    if fp and fp.get("production"):
        return fp
    detail = f" weight problems: {fp['weight_problems']}" if (fp or {}).get("weight_problems") else ""
    message = (
        f"{what} fingerprint is not production-grade (mode={(fp or {}).get('mode')!r}, "
        f"production={(fp or {}).get('production')!r}, path={(fp or {}).get('path')!r})."
        f"{detail} A real run must bind its readout cache and resume keys to checkpoint "
        "CONTENT, or an in-place weight overwrite would silently reuse stale logits."
    )
    if allow_nonproduction:
        return fp
    raise FingerprintError(message + " Pass --allow-nonproduction-fingerprint to "
                                     "override deliberately.")


def resolve(checkpoint_dir, dry_run=False, cache_dir=None, mode=MODE_CONTENT,
            allow_nonproduction=False, label=None, warn=None):
    """The single entry point callers use: dry-run identity or a verified fingerprint."""
    if dry_run:
        return dry_run_fingerprint(label or checkpoint_dir or "dry-run")
    fp = fingerprint(checkpoint_dir, mode=mode, cache_dir=cache_dir,
                     allow_missing=allow_nonproduction)
    require_production(fp, what=f"checkpoint {checkpoint_dir!r}",
                       allow_nonproduction=allow_nonproduction)
    if warn and not fp.get("production"):
        warn(f"--allow-nonproduction-fingerprint: {checkpoint_dir!r} has a "
             f"{fp.get('mode')} fingerprint; results are NOT content-addressed")
    return fp


def describe(fp):
    """The provenance block recorded next to every number a checkpoint produced."""
    return {
        "path": (fp or {}).get("path"),
        "mode": (fp or {}).get("mode"),
        "production": bool((fp or {}).get("production")),
        "fingerprint": (fp or {}).get("fingerprint"),
        "weights_digest": (fp or {}).get("weights_digest"),
        "meta_digest": (fp or {}).get("meta_digest"),
        "n_weight_files": (fp or {}).get("n_weight_files"),
        "total_weight_bytes": (fp or {}).get("total_weight_bytes"),
        "hash_source": (fp or {}).get("hash_source"),
    }


def matches(fp, recorded):
    """Whether a freshly computed fingerprint matches one recorded earlier."""
    if not fp or not recorded:
        return False
    return bool(fp.get("fingerprint")) and fp["fingerprint"] == recorded.get("fingerprint")
