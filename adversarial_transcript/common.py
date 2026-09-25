"""Stdlib-only JSON, transcript, stance, and atomic-write helpers.

Keeping these helpers dependency-light lets schema validation and offline selection run
without importing model clients or GPU libraries.
"""

import json
import os
import random
import tempfile

# Stance labels are exactly these two strings everywhere (mirrors debate.validate).
STANCE_LABELS = ("Y_true", "Y_false")
DEBATER_NAMES = ("Debater A", "Debater B")
STANCE_SEED = 1090998506967575423  # mirrors debate.SEED


# ---- IO -------------------------------------------------------------------
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    """Read a JSON-lines file into a list, skipping blank lines."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON line: {e}") from e
    return rows


def _atomic_write(path, render):
    """Write via a temp file in the destination dir then os.replace (atomic).

    Mirrors debate.atomic_write_json but stays generic (render writes to the fd's
    file object) so it serves both whole-JSON and JSONL outputs.
    """
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            render(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json(path, obj):
    _atomic_write(path, lambda f: json.dump(obj, f, ensure_ascii=False, indent=2))


def write_jsonl(path, rows):
    def render(f):
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    _atomic_write(path, render)


# ---- Dataset loading ------------------------------------------------------
def load_items(path):
    """Load the QuALITY-H dataset: a TOP-LEVEL JSON list of example dicts.

    Each example is {story_title, q_m:{question,Y_true,Y_false[,transcript]},
    q_h:{question,H_true,H_false}}. We validate only the shape we rely on and
    leave any extra keys untouched.
    """
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a top-level JSON list (QuALITY-H schema), "
            f"got {type(data).__name__}"
        )
    return data


def load_story_map(path):
    """Load the {story_title: story_text} map (dataset/title-story.json)."""
    sm = read_json(path)
    if not isinstance(sm, dict):
        raise ValueError(f"{path}: story map must be a JSON object {{title: story}}")
    return sm


def item_id_for(index):
    """Stable per-item id from its 0-based position in the dataset list."""
    return f"qh_{int(index):04d}"


# ---- Transcript helpers ---------------------------------------------------
def iter_round_args(transcript):
    """Yield every (debater_name, argument_text) across all rounds of a transcript."""
    for rnd in transcript.get("rounds", []):
        for name in DEBATER_NAMES:
            arg = rnd.get(name)
            if isinstance(arg, str):
                yield name, arg


def transcript_text(transcript):
    """Flatten all round arguments (both debaters) into one string for text scans."""
    return " ".join(arg for _, arg in iter_round_args(transcript))


def compromised_debater_from_stance(stance):
    """Return the debater NAME whose stance is 'Y_true' (the one to compromise).

    `stance` is {"Debater A": "Y_true|Y_false", "Debater B": ...}. Exactly one
    defends Y_true; return its name, or None if the labels are malformed.
    """
    for name in DEBATER_NAMES:
        if stance.get(name) == "Y_true":
            return name
    return None


def resolve_stance(example):
    """Resolve which debater defends Y_true vs Y_false for an example.

    Policy (constraint 7): PREFER the stance recorded in an existing honest
    transcript (QuALITY-H-with-honest-transcripts.json); else fall back to the
    deterministic stance used by debate.stance_rng, reproducing debate.debate's
    `a_defends_true = rng.random() < 0.5` so generation order / resume can't change it.

    Returns {"Debater A": label, "Debater B": label, "source": "honest_transcript"|"stance_rng"}.
    """
    t = example.get("Q_Y", {}).get("transcript")
    if isinstance(t, dict) and t.get("Debater A") in STANCE_LABELS and t.get("Debater B") in STANCE_LABELS:
        return {"Debater A": t["Debater A"], "Debater B": t["Debater B"], "source": "honest_transcript"}

    # Fallback: deterministic stance. This deliberately duplicates debate.stance_rng
    # instead of importing debate.py, keeping selection/tests dependency-free.
    q_y = example["Q_Y"]
    key = "|".join(
        str(part)
        for part in (
            STANCE_SEED,
            example.get("story_title"),
            q_y["question"],
            q_y["Y_true"],
            q_y["Y_false"],
        )
    )
    rng = random.Random(key)
    a_defends_true = rng.random() < 0.5  # MUST match debate.debate's stance draw
    return {
        "Debater A": "Y_true" if a_defends_true else "Y_false",
        "Debater B": "Y_false" if a_defends_true else "Y_true",
        "source": "stance_rng",
    }
