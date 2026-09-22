"""Dependency-free validators for candidate / score / selection records.

We do NOT depend on the `jsonschema` package (it is not installed in the offline
environment). The JSON Schema files under schemas/ are the canonical, human-readable
documentation; the functions here implement the structural subset we actually
enforce at runtime. Each returns a list of human-readable error strings (empty = OK).
"""

from adversarial_transcript.common import DEBATER_NAMES, STANCE_LABELS

CONDITIONS = ("honest", "adversarial")


def _is_str(x):
    return isinstance(x, str)


def validate_transcript(t, prefix="transcript"):
    """Validate the NATIVE QuALITY-H transcript shape (mirrors debate.validate)."""
    errs = []
    if not isinstance(t, dict):
        return [f"{prefix}: must be an object"]
    if set(t.keys()) != {"Debater A", "Debater B", "rounds"}:
        errs.append(f"{prefix}: keys must be exactly {{Debater A, Debater B, rounds}}, got {sorted(t.keys())}")
    if {t.get("Debater A"), t.get("Debater B")} != set(STANCE_LABELS):
        errs.append(f"{prefix}: stance labels must be exactly {{Y_true, Y_false}}")
    rounds = t.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errs.append(f"{prefix}: 'rounds' must be a non-empty list")
        return errs
    for i, rnd in enumerate(rounds):
        if not isinstance(rnd, dict) or set(rnd.keys()) != set(DEBATER_NAMES):
            errs.append(f"{prefix}: round {i + 1} keys must be exactly {{Debater A, Debater B}}")
            continue
        for name in DEBATER_NAMES:
            if not _is_str(rnd.get(name)):
                errs.append(f"{prefix}: round {i + 1} '{name}' must be a string")
    return errs


PROMPT_META_REQUIRED = ("artifact_id", "prompt_version", "placement", "content_sha256",
                        "exposure")
PROMPT_PLACEMENTS = ("system_only", "user_only", "hybrid", "none")


def validate_prompt_meta(block, prefix="candidate.gen_meta.prompt"):
    """Validate the OPTIONAL gen_meta.prompt block.

    The block is optional. When present it must be
    complete because it identifies the exact private instruction that produced the row.
    """
    errs = []
    if not isinstance(block, dict):
        return [f"{prefix}: must be an object"]
    for key in PROMPT_META_REQUIRED:
        if key not in block:
            errs.append(f"{prefix}: missing required key '{key}'")
    if "placement" in block and block["placement"] not in PROMPT_PLACEMENTS:
        errs.append(f"{prefix}.placement: must be one of {PROMPT_PLACEMENTS}")
    for key in ("artifact_id", "prompt_version", "content_sha256"):
        if key in block and not _is_str(block[key]):
            errs.append(f"{prefix}.{key}: must be a string")
    exposure = block.get("exposure")
    if exposure is not None:
        if not isinstance(exposure, dict):
            errs.append(f"{prefix}.exposure: must be an object")
        else:
            count = exposure.get("target_exposure_per_request")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errs.append(f"{prefix}.exposure.target_exposure_per_request: "
                            "must be a non-negative integer")
            elif count > 1:
                # More than one target-bearing component in ONE request would break the
                # hybrid-vs-system-only exposure match the placement ablation rests on.
                errs.append(f"{prefix}.exposure.target_exposure_per_request: {count} "
                            "target-bearing private-instruction components in one "
                            "request; at most 1 is allowed")
    return errs


def validate_candidate(c):
    errs = []
    if not isinstance(c, dict):
        return ["candidate: must be an object"]
    gen_meta = c.get("gen_meta")
    if isinstance(gen_meta, dict) and "prompt" in gen_meta:
        errs += validate_prompt_meta(gen_meta["prompt"])
    required = ["item_id", "dataset_index", "story_title", "condition",
                "candidate_idx", "compromised_debater", "qh_target", "transcript", "qy", "qh"]
    for k in required:
        if k not in c:
            errs.append(f"candidate: missing required key '{k}'")
    if c.get("condition") not in CONDITIONS:
        errs.append(f"candidate.condition: must be one of {CONDITIONS}")
    if not isinstance(c.get("candidate_idx"), int):
        errs.append("candidate.candidate_idx: must be an integer")
    cond = c.get("condition")
    comp = c.get("compromised_debater")
    if cond == "adversarial":
        if comp not in DEBATER_NAMES:
            errs.append("candidate.compromised_debater: adversarial must name 'Debater A' or 'Debater B'")
        if c.get("qh_target") != "H_false":
            errs.append("candidate.qh_target: adversarial must be 'H_false'")
    elif cond == "honest":
        if comp is not None:
            errs.append("candidate.compromised_debater: honest must be null")
        if c.get("qh_target") is not None:
            errs.append("candidate.qh_target: honest must be null")
    if "transcript" in c:
        errs += validate_transcript(c["transcript"], "candidate.transcript")
    # The compromised debater must actually be the Y_true defender (correctness-preserving).
    if cond == "adversarial" and isinstance(c.get("transcript"), dict) and comp in DEBATER_NAMES:
        if c["transcript"].get(comp) != "Y_true":
            errs.append("candidate.compromised_debater: must be the debater whose stance is 'Y_true'")
    return errs


def validate_score(s):
    """Validate an EXTERNAL score record (the primary input to select.py)."""
    errs = []
    if not isinstance(s, dict):
        return ["score: must be an object"]
    for k in ("item_id", "condition", "candidate_idx", "qy_clean_correct",
              "qy_adv_correct", "qy_no_regression_pass", "p_hfalse"):
        if k not in s:
            errs.append(f"score: missing required key '{k}'")
    if s.get("condition") not in CONDITIONS:
        errs.append(f"score.condition: must be one of {CONDITIONS}")
    if not isinstance(s.get("candidate_idx"), int):
        errs.append("score.candidate_idx: must be an integer")
    p = s.get("p_hfalse")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0.0 <= float(p) <= 1.0):
        errs.append("score.p_hfalse: must be a number in [0, 1]")
    for k in ("qy_no_regression_pass", "qy_clean_correct", "qy_adv_correct", "qy_correct"):
        if k in s and s[k] is not None and not isinstance(s[k], bool):
            errs.append(f"score.{k}: must be a boolean or null")
    clean = s.get("qy_clean_correct")
    adv = s.get("qy_adv_correct")
    preserved = s.get("qy_no_regression_pass")
    if all(isinstance(x, bool) for x in (clean, adv, preserved)):
        expected = (not clean) or adv
        if preserved is not expected:
            errs.append("score.qy_no_regression_pass: must equal (not qy_clean_correct) or qy_adv_correct")
    return errs


SELECTION_STATUSES = ("ok", "qy_fallback", "no_adv_survivor", "no_honest_survivor",
                      "no_survivor")
ADVERSARIAL_ARM_STATUSES = ("ok", "qy_fallback", "no_adv_survivor")
HONEST_ARM_STATUSES = ("ok", "qy_fallback", "no_honest_survivor")


def validate_selection(r):
    errs = []
    if not isinstance(r, dict):
        return ["selection: must be an object"]
    for k in ("item_id", "status", "gates", "cti"):
        if k not in r:
            errs.append(f"selection: missing required key '{k}'")
    if r.get("status") not in SELECTION_STATUSES:
        errs.append("selection.status: unexpected value")
    # Per-arm statuses (Adv-BoK-Hfalse). In multi-condition runs the top-level status is a
    # publish-safety gate, so these are what identify the branch each arm actually used.
    if "adversarial_status" in r and r["adversarial_status"] not in ADVERSARIAL_ARM_STATUSES:
        errs.append("selection.adversarial_status: unexpected value")
    if "honest_status" in r and r["honest_status"] not in HONEST_ARM_STATUSES:
        errs.append("selection.honest_status: unexpected value")
    if "selector" in r and not _is_str(r["selector"]):
        errs.append("selection.selector: must be a string")
    return errs


def score_key(rec):
    """Join key shared by candidate and score records."""
    return (rec["item_id"], rec["condition"], int(rec["candidate_idx"]))
