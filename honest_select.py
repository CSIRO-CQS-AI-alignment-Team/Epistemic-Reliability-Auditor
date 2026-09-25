"""Score and select the honest best-of-K debate pool, then publish the dataset files.

Stages (split like the adversarial arm, because only `score` needs a GPU):

  --stage score   runs/honest/<DS>/candidate.jsonl -> verifier_scores.jsonl
  --stage select  candidate.jsonl + verifier_scores.jsonl -> selected.jsonl + report.json
  --stage write   selected.jsonl -> dataset/<DS>/<DS>.json (updated in place)
                                 -> dataset/<DS>/<DS>-with-honest-transcripts.json
  --stage all     score -> select -> write

Selector: H-BoK-Htrue.

  1. q_m gate: the UNFINE-TUNED base verifier (gpt-oss-20b), having read the honest
     transcript, must answer Y_true.
  2. Among the candidates that pass, take argmax p(H_true).
  3. If ALL K fail the q_m gate, fall back to argmax p(H_true) over EVERY validly
     scored candidate of that item -- the full pool, never a filtered subset.

There is no hygiene/leakage/quote gate: leakage and quote counts are recorded in the
audits for inspection, and nothing but the q_m gate can remove a candidate from the
ranking. That keeps step 3 exactly as specified.

What every stage re-verifies before touching anything (see honest_bok_io):

  * run manifest identity + per-candidate `gen_meta.manifest_digest`, so a dry-run
    or mismatched model, seed, prompt, or input cannot be scored, selected, or published;
  * best-of-K completeness: every item carries exactly candidate_idx 0..K-1;
  * base-verifier provenance for EVERY candidate's score -- real forced-choice scorer,
    matching model, option seed, prompt version, and all three prompt hashes;
  * candidate<->score join keys, and dataset index alignment across the three files.

Two directional traps this file exists to avoid:

  * `adversarial_transcript.filters.correctness` implements the NO-REGRESSION rule
    ((not clean) or adv). The honest selector needs ABSOLUTE transcript-conditioned
    correctness, so the gate is implemented here instead of reused.
  * `adversarial_transcript.filters.cti.rank` maximises p_hfalse. The honest arm
    maximises p_htrue -- the OPPOSITE direction.

Everything else is reused read-only from `adversarial_transcript/`, which is never
modified.
"""

import argparse
import copy
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import honest_bok_io as hio  # noqa: E402
from adversarial_transcript import schema, score_verifier  # noqa: E402
from adversarial_transcript.filters import leakage, quote_audit  # noqa: E402

SELECTOR = "h-bok-htrue"
# Extra provenance we stamp into each score row's `meta`. score_verifier.py is
# read-only, so these are added after it returns; the score schema allows extra keys.
SCORE_FINGERPRINT_FIELD = "honest_bok_verifier_fingerprint"
SCORE_POOL_DIGEST_FIELD = "honest_bok_pool_manifest_digest"
SCORE_VERIFIER_ID_FIELD = "honest_bok_verifier_id"
STATUS_OK = hio.STATUS_OK
STATUS_FALLBACK = hio.STATUS_FALLBACK
STATUS_NONE = hio.STATUS_NONE


class SelectConfig:
    """Everything the score/select/write stages are pinned to."""

    def __init__(self, ws, k, num_rounds, model, device, option_seed, dry_run,
                 start_row=0, limit=None, rehash_verifier=False):
        self.ws = ws
        self.k = int(k)
        self.num_rounds = int(num_rounds)
        self.model = model
        self.device = int(device)
        self.option_seed = int(option_seed)
        self.dry_run = bool(dry_run)
        self.start_row = int(start_row)
        self.limit = limit
        # Bypass the fingerprint cache entirely; the full-content hash is authoritative.
        self.rehash_verifier = bool(rehash_verifier)


# ---- shared verification ---------------------------------------------------
def load_candidates(path):
    """Strictly load candidate.jsonl. Any defect aborts -- nothing is skipped or deduped.

    Downstream of generation there is no such thing as a tolerable bad row: a malformed
    line, a foreign condition, or a duplicate join key makes the pool ambiguous, and
    "normalising" it would decide by accident which transcript gets published.
    """
    _, by_key = hio.load_rows_strict(path, "candidates", schema.validate_candidate,
                                     hio.candidate_key)
    return list(by_key.values()), by_key


def load_scores(path):
    """Strictly load verifier_scores.jsonl under the same zero-tolerance rule."""
    _, by_key = hio.load_rows_strict(path, "scores", schema.validate_score,
                                     hio.candidate_key)
    return by_key


def verify_pool(ws, config, stage, warn=hio.stderr_warn, require_scores=True):
    """Load and fully verify the candidate pool (+ scores). Returns a bundle dict.

    Raises HonestBokError on the first violation: this is the gate that stops a
    contaminated or partial pool from reaching selection or the canonical dataset.
    """
    manifest = hio.require_manifest_for_read(ws, config.dry_run, stage)
    identity = manifest["identity"]
    digest = manifest["identity_digest"]

    if int(identity.get("k", -1)) != config.k:
        raise hio.HonestBokError(
            "%s: manifest was generated with k=%s but this run uses --k %d; the "
            "completeness check would be meaningless."
            % (stage, identity.get("k"), config.k))
    if int(identity.get("num_rounds", -1)) != config.num_rounds:
        raise hio.HonestBokError(
            "%s: manifest num_rounds=%s but this run uses --num-rounds %d"
            % (stage, identity.get("num_rounds"), config.num_rounds))

    # The scores must come from the declared, recognised, unfine-tuned base verifier,
    # and must be the exact artifact the score stage sealed.
    verifier_manifest = hio.require_verifier_manifest(
        ws, config.model, config.option_seed, score_verifier.PROMPT_VERSION,
        config.dry_run, stage, require_sealed=require_scores,
        rehash=config.rehash_verifier)

    items = hio.load_dataset_items(ws.items)
    hio.check_items_are_no_debate(items, ws.items)
    stories = hio.load_story_map(ws.stories)
    candidates, candidates_by_key = load_candidates(ws.candidates)

    for candidate in candidates:
        reason = hio.check_candidate_provenance(candidate, digest, identity)
        if reason is not None:
            raise hio.HonestBokError(
                "%s: candidate %s#%s does not belong to this run (%s). Regenerate the "
                "pool; a mixed pool must never be selected from."
                % (stage, candidate.get("item_id"), candidate.get("candidate_idx"), reason))
        if candidate.get("story_title") not in stories:
            raise hio.HonestBokError(
                "%s: candidate %s references a story missing from %s"
                % (stage, candidate.get("item_id"), ws.stories))

    hio.check_pool_complete(items, candidates, config.k, stage)

    scores_by_key = {}
    if require_scores:
        scores_by_key = load_scores(ws.scores)
        missing = [key for key in candidates_by_key if key not in scores_by_key]
        if missing:
            raise hio.HonestBokError(
                "%s: %d candidate(s) have no verifier score (first: %s); score the full "
                "pool before selecting." % (stage, len(missing), missing[:5]))
        foreign = [key for key in scores_by_key if key not in candidates_by_key]
        if foreign:
            raise hio.HonestBokError(
                "%s: %d score row(s) do not join to any candidate (first: %s)"
                % (stage, len(foreign), foreign[:5]))

        check_scorer_uniformity(scores_by_key, stage)
        for key, row in scores_by_key.items():
            check_score_provenance(row, candidates_by_key[key], stories, config, stage)
            check_score_binding(row, verifier_manifest.get("fingerprint"), digest,
                                verifier_manifest.get("verifier_id"), stage)

    return {"manifest": manifest, "identity": identity, "digest": digest,
            "verifier_manifest": verifier_manifest,
            "items": items, "stories": stories, "candidates": candidates,
            "candidates_by_key": candidates_by_key, "scores_by_key": scores_by_key}


def _round_eq(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _require_prob(value, label, key, stage):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise hio.HonestBokError("%s: score %r %s is not a number" % (stage, key, label))
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise hio.HonestBokError("%s: score %r %s=%r is outside [0, 1]"
                                 % (stage, key, label, value))
    return value


def _check_prob_block(row, field, order, semantic_labels, key, stage):
    """Validate one `_rounded_probs`-shaped block: {letter: {A,B}, semantic: {...}}.

    Returns (letter_probs, argmax_letter). The semantic map is fully redundant with the
    letter map under the recorded answer order, so it is RECOMPUTED rather than trusted.
    """
    block = row.get(field)
    if not isinstance(block, dict):
        raise hio.HonestBokError("%s: score %r has no %s block" % (stage, key, field))
    letters = block.get("letter")
    semantic = block.get("semantic")
    if not isinstance(letters, dict) or set(letters) != {"A", "B"}:
        raise hio.HonestBokError("%s: score %r %s.letter is malformed"
                                 % (stage, key, field))
    if not isinstance(semantic, dict) or set(semantic) != set(semantic_labels):
        raise hio.HonestBokError(
            "%s: score %r %s.semantic must have exactly the keys %s, got %s"
            % (stage, key, field, sorted(semantic_labels), sorted(semantic or {})))
    for letter in ("A", "B"):
        _require_prob(letters[letter], "%s.letter.%s" % (field, letter), key, stage)
    if not _round_eq(letters["A"] + letters["B"], 1.0):
        raise hio.HonestBokError("%s: score %r %s.letter does not sum to 1"
                                 % (stage, key, field))
    for letter, label in order.items():
        if not _round_eq(semantic.get(label, -2), letters[letter]):
            raise hio.HonestBokError(
                "%s: score %r %s.semantic[%r] does not equal %s.letter[%r] under the "
                "recorded answer order" % (stage, key, field, label, field, letter))
    return letters, max(letters, key=letters.get)


def _check_meta_letter_copy(meta, meta_field, letters, key, stage):
    """meta duplicates the letter probabilities; the copy must match the original."""
    copy_block = meta.get(meta_field)
    if not isinstance(copy_block, dict) or set(copy_block) != {"A", "B"}:
        raise hio.HonestBokError("%s: score %r meta.%s is malformed"
                                 % (stage, key, meta_field))
    for letter in ("A", "B"):
        if not _round_eq(copy_block[letter], letters[letter]):
            raise hio.HonestBokError(
                "%s: score %r meta.%s[%r]=%r disagrees with the top-level letter "
                "probabilities (%r)" % (stage, key, meta_field, letter,
                                        copy_block[letter], letters[letter]))


def check_score_consistency(row, candidate, config, stage):
    """Recompute EVERY deterministic redundancy score_candidate emits.

    `score_candidate` writes each quantity several times over: letter and semantic
    probability maps, predicted letters and semantic labels, correctness flags, compact
    aliases, and a second copy of both q_m letter maps inside `meta`. Every
    one of those is derivable from (letter probabilities, deterministic answer order), so
    every one is recomputed here rather than trusted. A row that is schema-valid but
    internally inconsistent -- p_htrue edited while p_hfalse is left alone, a flipped
    `qh_probs.pred`, a doctored `qy_adv_probs.semantic` -- can change which candidate the
    selector picks, or misrepresent what the verifier actually answered.

    Nothing here re-runs the model: it proves the row is self-consistent and matches the
    deterministic option order for THIS candidate.

    HONEST LIMIT, stated so the docstring matches the behaviour: three values are raw
    model measurements rather than derivable duplicates, so they cannot be recomputed
    offline --
      * `meta.readout_max_letter_logprob.{qy_clean,qy,qh}` (the readout-health signal).
    They are bounded here (exactly those three keys, each a number that cannot exceed 0
    because it is a log-probability) and they feed no selection decision; they are copied
    into selected.jsonl purely for auditing. `meta.readout` and
    `meta.model_quantized_load` are likewise not recomputable per row, and are instead
    pinned pool-wide by `check_scorer_uniformity`. Everything else score_candidate emits
    IS recomputed here.
    """
    key = hio.candidate_key(row)
    meta = row["meta"]
    if int(meta.get("option_seed", -1)) != int(config.option_seed):
        raise hio.HonestBokError(
            "%s: score %r meta.option_seed=%r but this run uses %r"
            % (stage, key, meta.get("option_seed"), config.option_seed))

    # --- q_h: the ranked quantity ---
    if "p_htrue" not in row or row.get("p_htrue") is None:
        raise hio.HonestBokError(
            "%s: score %r has no p_htrue. Production selection ranks on p(H_true) and "
            "will not reconstruct it from 1 - p_hfalse; re-score the pool." % (stage, key))
    p_htrue = _require_prob(row.get("p_htrue"), "p_htrue", key, stage)
    p_hfalse = _require_prob(row.get("p_hfalse"), "p_hfalse", key, stage)
    if not _round_eq(p_htrue + p_hfalse, 1.0):
        raise hio.HonestBokError(
            "%s: score %r has p_htrue + p_hfalse = %.8f, not 1. The forced choice is "
            "over exactly two options, so this row is internally inconsistent."
            % (stage, key, p_htrue + p_hfalse))

    qh_order = score_verifier.deterministic_answer_order(
        candidate, "Q_H", config.option_seed)
    if meta.get("qh_answer_order") != qh_order:
        raise hio.HonestBokError(
            "%s: score %r meta.qh_answer_order %r != the deterministic order %r"
            % (stage, key, meta.get("qh_answer_order"), qh_order))
    qh_letters, qh_argmax = _check_prob_block(
        row, "qh_probs", qh_order, ("H_true", "H_false"), key, stage)
    qh_semantic = row["qh_probs"]["semantic"]
    if not _round_eq(qh_semantic["H_true"], p_htrue) or \
            not _round_eq(qh_semantic["H_false"], p_hfalse):
        raise hio.HonestBokError(
            "%s: score %r p_htrue/p_hfalse disagree with qh_probs.semantic"
            % (stage, key))
    if row["qh_probs"].get("pred_letter") != qh_argmax:
        raise hio.HonestBokError(
            "%s: score %r qh_probs.pred_letter=%r but qh_probs.letter argmax is %r"
            % (stage, key, row["qh_probs"].get("pred_letter"), qh_argmax))
    if row["qh_probs"].get("pred") != qh_order[qh_argmax]:
        raise hio.HonestBokError(
            "%s: score %r qh_probs.pred=%r but the predicted letter %r maps to %r"
            % (stage, key, row["qh_probs"].get("pred"), qh_argmax, qh_order[qh_argmax]))

    # --- q_m: the gate ---
    qy_order = score_verifier.deterministic_answer_order(
        candidate, "Q_Y", config.option_seed)
    if meta.get("qy_answer_order") != qy_order:
        raise hio.HonestBokError(
            "%s: score %r meta.qy_answer_order %r != the deterministic order %r"
            % (stage, key, meta.get("qy_answer_order"), qy_order))
    gold_letter = next(letter for letter, label in qy_order.items() if label == "Y_true")

    for field, correct_field, pred_letter_field, pred_field, meta_field in (
        ("qy_adv_probs", "qy_adv_correct", "qy_adv_pred_letter", "qy_adv_pred",
         "qy_letter_probs"),
        ("qy_clean_probs", "qy_clean_correct", "qy_clean_pred_letter", "qy_clean_pred",
         "qy_clean_letter_probs"),
    ):
        letters, predicted = _check_prob_block(
            row, field, qy_order, ("Y_true", "Y_false"), key, stage)
        if row.get(pred_letter_field) != predicted:
            raise hio.HonestBokError(
                "%s: score %r %s=%r but %s argmax is %r"
                % (stage, key, pred_letter_field, row.get(pred_letter_field), field,
                   predicted))
        if row.get(pred_field) != qy_order[predicted]:
            raise hio.HonestBokError(
                "%s: score %r %s=%r but the predicted letter %r maps to %r"
                % (stage, key, pred_field, row.get(pred_field), predicted,
                   qy_order[predicted]))
        expected_correct = predicted == gold_letter
        if row.get(correct_field) is not expected_correct:
            raise hio.HonestBokError(
                "%s: score %r %s=%r but the readout chose %r (gold %r)"
                % (stage, key, correct_field, row.get(correct_field), predicted,
                   gold_letter))
        _check_meta_letter_copy(meta, meta_field, letters, key, stage)

    # --- fields that are not recomputable but are still bounded ---
    baseline = row.get("baseline_p_hfalse")
    if baseline is not None:
        _require_prob(baseline, "baseline_p_hfalse", key, stage)
    health = meta.get("readout_max_letter_logprob")
    if health is not None:
        if not isinstance(health, dict) or set(health) != {"qy_clean", "qy", "qh"}:
            raise hio.HonestBokError(
                "%s: score %r meta.readout_max_letter_logprob must have exactly the keys "
                "qy_clean/qy/qh, got %s" % (stage, key, sorted(health or {})))
        for name, value in health.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise hio.HonestBokError(
                    "%s: score %r meta.readout_max_letter_logprob.%s is not a number"
                    % (stage, key, name))
            # It is max_l log P(letter): a log-probability, so it cannot exceed 0.
            if float(value) > 1e-6:
                raise hio.HonestBokError(
                    "%s: score %r meta.readout_max_letter_logprob.%s=%r is positive, "
                    "which is impossible for a log-probability"
                    % (stage, key, name, value))

    clean, adv = row.get("qy_clean_correct"), row.get("qy_adv_correct")
    if row.get("qy_no_regression_pass") is not ((not clean) or adv):
        raise hio.HonestBokError(
            "%s: score %r qy_no_regression_pass disagrees with its own flags"
            % (stage, key))
    # Schema aliases must agree with their explicit q_m fields.
    for alias, canonical in (("qy_correct", "qy_adv_correct"),
                             ("qy_preserved", "qy_no_regression_pass"),
                             ("qy_pred_letter", "qy_adv_pred_letter")):
        if alias in row and row[alias] is not None and row[alias] != row.get(canonical):
            raise hio.HonestBokError("%s: score %r alias %s disagrees with %s"
                                     % (stage, key, alias, canonical))
    return p_htrue


def check_scorer_uniformity(scores_by_key, stage):
    """Scorer-describing metadata must be identical across the whole pool.

    `meta.readout` and `meta.model_quantized_load` describe HOW the verifier was read,
    not what it answered, so they cannot be recomputed from a single row -- but every
    row in a pool comes from one scorer, so they must agree. Cross-row equality catches
    per-row tampering without hardcoding adversarial_transcript's private constants.
    """
    seen = {}
    for key, row in sorted(scores_by_key.items()):
        meta = row.get("meta") or {}
        signature = (meta.get("readout"), meta.get("model_quantized_load"))
        seen.setdefault(signature, []).append(key)
    if len(seen) > 1:
        raise hio.HonestBokError(
            "%s: the scores were not all produced by one scorer configuration -- "
            "meta.readout/model_quantized_load differ across rows: %s"
            % (stage, {str(sig): keys[:2] for sig, keys in seen.items()}))
    (readout, _quantized), = seen.keys()
    if not isinstance(readout, str) or not readout:
        raise hio.HonestBokError("%s: meta.readout is missing or not a string" % stage)


def check_score_provenance(row, candidate, stories, config, stage):
    """The score must be a REAL base-verifier readout of THIS candidate."""
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise hio.HonestBokError("%s: score %r has no meta block"
                                 % (stage, hio.candidate_key(row)))
    scorer = row.get("scorer")
    expected_scorer = "dry-run-forced-choice" if config.dry_run else hio.PRODUCTION_SCORER
    if scorer != expected_scorer:
        raise hio.HonestBokError(
            "%s: score %r came from scorer %r but this run requires %r"
            % (stage, hio.candidate_key(row), scorer, expected_scorer))
    expected_model = "dry-run" if config.dry_run else config.model
    if meta.get("model") != expected_model:
        raise hio.HonestBokError(
            "%s: score %r was produced by verifier %r, not %r"
            % (stage, hio.candidate_key(row), meta.get("model"), expected_model))
    story = stories[candidate["story_title"]]
    # Reuses the adversarial staleness check: prompt_version + model + option_seed +
    # all three rendered prompt hashes must match a recomputation from the candidate.
    if not score_verifier.score_matches_current_run(
        row, candidate, story, config.model, config.option_seed, config.dry_run
    ):
        raise hio.HonestBokError(
            "%s: score %r does not match a recomputation of its candidate's prompts "
            "(stale prompt_version/option_seed/model, or the transcript changed)"
            % (stage, hio.candidate_key(row)))
    check_score_consistency(row, candidate, config, stage)


def check_score_binding(row, expected_fingerprint, expected_pool_digest,
                        expected_verifier_id, stage):
    """The row must name the checkpoint it was scored under, and this pool.

    ``meta.model`` records only a path. Different checkpoints can occupy that path, so
    the content fingerprint prevents stored logits from being attributed to another
    model.
    """
    key = hio.candidate_key(row)
    meta = row.get("meta") or {}
    if SCORE_FINGERPRINT_FIELD not in meta:
        raise hio.HonestBokError(
            "%s: score %r predates checkpoint binding (no meta.%s), so the checkpoint "
            "that produced it cannot be identified. Re-score with --overwrite."
            % (stage, key, SCORE_FINGERPRINT_FIELD))
    recorded = meta.get(SCORE_FINGERPRINT_FIELD)
    # In the dry-run sandbox there is no checkpoint, so the recorded value is null and
    # only the equality below applies. Production additionally requires it to be real,
    # which require_verifier_manifest already enforces on the manifest side.
    if expected_fingerprint is not None and not recorded:
        raise hio.HonestBokError(
            "%s: score %r carries a null checkpoint fingerprint but the run has one"
            % (stage, key))
    if recorded != expected_fingerprint:
        raise hio.HonestBokError(
            "%s: score %r was produced under checkpoint fingerprint %s, but the requested "
            "checkpoint is %s. Re-score; logits must not be attributed to another "
            "verifier." % (stage, key, str(recorded)[:16], str(expected_fingerprint)[:16]))
    pool = meta.get(SCORE_POOL_DIGEST_FIELD)
    if pool != expected_pool_digest:
        raise hio.HonestBokError(
            "%s: score %r was produced for candidate pool %s, not %s"
            % (stage, key, str(pool)[:16], str(expected_pool_digest)[:16]))
    if meta.get(SCORE_VERIFIER_ID_FIELD) != expected_verifier_id:
        raise hio.HonestBokError(
            "%s: score %r records verifier id %r, not %r"
            % (stage, key, meta.get(SCORE_VERIFIER_ID_FIELD), expected_verifier_id))


# ---- score stage ----------------------------------------------------------
def run_score_stage(ws, config, overwrite=False, warn=hio.stderr_warn):
    manifest = hio.require_manifest_for_read(ws, config.dry_run, "score")
    identity = manifest["identity"]
    digest = manifest["identity_digest"]
    if int(identity.get("k", -1)) != config.k:
        raise hio.HonestBokError("score: manifest k=%s but --k %d"
                                 % (identity.get("k"), config.k))

    items = hio.load_dataset_items(ws.items)
    hio.check_items_are_no_debate(items, ws.items)
    stories = hio.load_story_map(ws.stories)
    # Strict here too: the production score entry must not consume a dirty pool either.
    candidates, by_key = load_candidates(ws.candidates)
    for candidate in candidates:
        reason = hio.check_candidate_provenance(candidate, digest, identity)
        if reason is not None:
            raise hio.HonestBokError(
                "score: candidate %s#%s does not belong to this run (%s)"
                % (candidate.get("item_id"), candidate.get("candidate_idx"), reason))
        if candidate.get("story_title") not in stories:
            raise hio.HonestBokError("score: candidate %s references a missing story"
                                     % candidate.get("item_id"))
    # Require complete best-of-K coverage before loading the verifier so a successful
    # score stage always yields a select-ready artifact.
    hio.check_pool_complete(items, candidates, config.k, "score")

    # Record (and, for production, validate) the base-verifier identity BEFORE scoring,
    # so select/write can re-verify that these scores came from the un-fine-tuned base.
    verifier_manifest = hio.build_verifier_manifest(
        config.model, config.option_seed, score_verifier.PROMPT_VERSION, config.dry_run,
        require_fingerprint=not config.dry_run, cache_dir=ws.run_dir,
        force=config.rehash_verifier)
    fingerprint = verifier_manifest["fingerprint"]
    hio.atomic_write_json(ws.verifier_manifest, verifier_manifest)
    print("[honest-select/score] verifier=%r id=%r fingerprint=%s"
          % (config.model, verifier_manifest["verifier_id"],
             (fingerprint[:16] + "...") if fingerprint else "UNAVAILABLE"))

    selected = score_verifier.select_candidate_rows(
        candidates, start_row=config.start_row, limit=config.limit)

    if overwrite and os.path.exists(ws.scores):
        os.remove(ws.scores)

    existing, _ = hio.read_jsonl_resume(ws.scores, warn=warn)
    reusable = []
    for position, row in enumerate(existing, 1):
        # Structural defects are NOT resume damage -- we only ever append complete,
        # schema-valid score rows, so anything malformed means the artifact was
        # corrupted or hand-edited. Fail closed rather than quietly rewriting it away.
        errors = schema.validate_score(row)
        if errors:
            raise hio.HonestBokError(
                "score: %s row %d is not a valid score record (%s); refusing to "
                "silently drop it. Re-run with --overwrite to rescore from scratch."
                % (ws.scores, position, errors[0]))
        try:
            key = hio.candidate_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise hio.HonestBokError("score: %s row %d has an unusable join key (%s)"
                                     % (ws.scores, position, exc))
        candidate = by_key.get(key)
        if candidate is None:
            raise hio.HonestBokError(
                "score: %s row %d (%r) does not join to any candidate in the pool"
                % (ws.scores, position, key))
        # A schema-valid row with mismatched model, prompt, option seed, or checkpoint
        # provenance is re-scored. The checkpoint fingerprint prevents logits from one
        # set of weights being attributed to another verifier at the same path.
        try:
            check_score_provenance(row, candidate, stories, config, "score")
            check_score_binding(row, fingerprint, digest,
                                verifier_manifest["verifier_id"], "score")
        except hio.HonestBokError as exc:
            warn("re-scoring %r: %s" % (key, exc))
            continue
        reusable.append(row)
    reusable, _ = hio.dedupe_identical_or_fail(reusable, hio.candidate_key, ws.scores)
    done = {hio.candidate_key(row) for row in reusable}

    # Rewrite with only the reusable rows so stale rows cannot survive.
    hio.atomic_write_jsonl(ws.scores, reusable)
    pending = [c for c in selected if hio.candidate_key(c) not in done]
    print("[honest-select/score] mode=%s candidates=%d reusable=%d pending=%d"
          % (ws.mode, len(candidates), len(reusable), len(pending)))

    if pending:
        scorer = None if config.dry_run else score_verifier.ForcedChoiceVerifier(
            config.model, config.device)
        with hio.JsonlAppender(ws.scores) as appender:
            for i, candidate in enumerate(pending, 1):
                story = stories[candidate["story_title"]]
                row = score_verifier.score_candidate(
                    candidate, story, scorer=scorer, option_seed=config.option_seed,
                    dry_run=config.dry_run)
                # Bind the row to the checkpoint it was actually scored under and to
                # this candidate pool. score_verifier.py is read-only, so we stamp the
                # provenance here; the score schema permits extra `meta` keys.
                row["meta"][SCORE_FINGERPRINT_FIELD] = fingerprint
                row["meta"][SCORE_VERIFIER_ID_FIELD] = verifier_manifest["verifier_id"]
                row["meta"][SCORE_POOL_DIGEST_FIELD] = digest
                appender.append(row)
                if i % 25 == 0 or i == len(pending):
                    print("[honest-select/score] %d/%d" % (i, len(pending)))

    _, stats = hio.canonicalize_jsonl(ws.scores, hio.candidate_key, hio.score_sort_key,
                                      warn=warn)
    # Seal the manifest to the FINAL artifact, then run the real selection-time gate so
    # that "score returned 0" provably means "a complete, select-ready scores file".
    hio.seal_verifier_manifest(ws, verifier_manifest)
    verify_pool(ws, config, "score", warn)
    print("[honest-select/score] scored=%d total=%d sealed -> %s"
          % (len(pending), stats["final_rows"], ws.scores))
    if config.dry_run:
        print("[honest-select/score] NOTE: dry-run scores are deterministic fakes in an "
              "isolated sandbox; they can never be published to the canonical dataset.")
    return 0


# ---- selection primitives --------------------------------------------------
def p_htrue_of(score):
    """p(H_true) for ranking. Requires the field: no 1 - p_hfalse reconstruction.

    Reconstructing it would mask exactly the internal inconsistency
    check_score_consistency exists to catch, and every row reaching selection has been
    validated to carry a coherent p_htrue.
    """
    if score is None:
        return None
    value = score.get("p_htrue")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        return None
    return value


def qy_true_gate(score):
    """H-BoK-Htrue gate: transcript-conditioned q_m readout must be Y_true.

    NOT the adversarial no-regression rule. Fail-closed on missing/ill-typed fields.
    `qy_adv_correct` is the transcript-conditioned flag; the "adv" prefix is the shared
    score schema's field name, not a statement about this arm.
    """
    if score is None:
        return False, "no_score"
    value = score.get("qy_adv_correct", score.get("qy_correct"))
    if value is None:
        return False, "qy_adv_correct_missing"
    if not isinstance(value, bool):
        return False, "qy_adv_correct_not_boolean"
    return value, ("pass" if value else "qy_not_true")


def rank_by_p_htrue(scored):
    """argmax p(H_true); ties broken toward the smaller candidate_idx.

    Direction is the OPPOSITE of adversarial_transcript.filters.cti.rank (max p_hfalse).
    """
    if not scored:
        return None
    return max(scored, key=lambda entry: (entry[2], -int(entry[0]["candidate_idx"])))


def _audit_entry(candidate, score, story, p_htrue, gate_reason):
    """Audit-only observations. Nothing here can remove a candidate from the ranking."""
    _, leak_info = leakage.check(candidate)
    if story is None:
        quote_info = {"skipped": True, "reason": "no_story_text"}
    else:
        quote_info = quote_audit.audit(candidate, story)
        quote_info["skipped"] = False
    return {
        "candidate_idx": candidate["candidate_idx"],
        "p_htrue": p_htrue,
        "p_hfalse": (score or {}).get("p_hfalse"),
        "qy_clean_correct": (score or {}).get("qy_clean_correct"),
        "qy_adv_correct": (score or {}).get("qy_adv_correct"),
        "qy_gate": gate_reason,
        "leakage": leak_info,
        "quote_audit": quote_info,
    }


def select_item(candidates, scores_index, story):
    """Run H-BoK-Htrue over one item's K candidates."""
    counts = {"total": 0, "no_score": 0, "drop_qy_gate": 0, "survivors": 0,
              "fallback_used": False}
    gated = []
    scored_all = []
    audits = []

    for candidate in sorted(candidates, key=lambda c: int(c["candidate_idx"])):
        counts["total"] += 1
        score = scores_index.get(schema.score_key(candidate))
        p_htrue = p_htrue_of(score)
        gate_passed, gate_reason = qy_true_gate(score)
        audits.append(_audit_entry(candidate, score, story, p_htrue, gate_reason))
        if score is None or p_htrue is None:
            counts["no_score"] += 1
            continue
        entry = (candidate, score, p_htrue)
        scored_all.append(entry)
        if not gate_passed:
            counts["drop_qy_gate"] += 1
            continue
        counts["survivors"] += 1
        gated.append(entry)

    best = rank_by_p_htrue(gated)
    status = STATUS_OK
    if best is None:
        # Spec'd fallback: no candidate made the verifier answer Y_true, so rank over
        # the FULL validly scored pool with no gate of any kind.
        best = rank_by_p_htrue(scored_all)
        if best is not None:
            status, counts["fallback_used"] = STATUS_FALLBACK, True
        else:
            status = STATUS_NONE
    return best, status, counts, audits


def build_selection_row(dataset_index, example, best, status, counts, audits):
    selected_candidate = selected_score = None
    p_htrue = None
    if best is not None:
        selected_candidate, selected_score, p_htrue = best
    joined = None
    if selected_candidate is not None:
        joined = dict(selected_candidate)
        joined["score"] = selected_score
    return {
        "schema_version": hio.SELECTION_SCHEMA_VERSION,
        "item_id": hio.common.item_id_for(dataset_index),
        "dataset_index": dataset_index,
        "story_title": example.get("story_title"),
        "status": status,
        "selector": SELECTOR,
        "honest_selected": joined,
        "gates": counts,
        "audits": audits,
        "readout": {
            "honest_p_htrue": p_htrue,
            "honest_p_hfalse": (selected_score or {}).get("p_hfalse"),
            "baseline_p_hfalse": (selected_score or {}).get("baseline_p_hfalse"),
            "scorer": (selected_score or {}).get("scorer"),
            "readout_max_letter_logprob": (
                ((selected_score or {}).get("meta") or {}).get("readout_max_letter_logprob")),
        },
    }


def process(bundle, config):
    """Pure selection over the FULL dataset; returns (selections, report)."""
    items = bundle["items"]
    scores_index = bundle["scores_by_key"]
    story_map = bundle["stories"]

    by_index = {}
    for candidate in bundle["candidates"]:
        by_index.setdefault(int(candidate["dataset_index"]), []).append(candidate)

    selections = []
    agg = {"total": 0, "no_score": 0, "drop_qy_gate": 0, "survivors": 0}
    status_counts = {STATUS_OK: 0, STATUS_FALLBACK: 0, STATUS_NONE: 0}
    p_htrue_selected = []
    fallback_items = []

    for dataset_index, example in enumerate(items):
        group = by_index.get(dataset_index, [])
        story = story_map.get(example.get("story_title"))
        best, status, counts, audits = select_item(group, scores_index, story)
        row = build_selection_row(dataset_index, example, best, status, counts, audits)

        errors = hio.validate_honest_selection(row, k=config.k)
        if errors:
            raise hio.HonestBokError("select: produced an invalid selection row for "
                                     "index %d: %s" % (dataset_index, errors[0]))
        selections.append(row)
        status_counts[status] += 1
        if status == STATUS_FALLBACK:
            fallback_items.append(dataset_index)
        if best is not None:
            p_htrue_selected.append(best[2])
        for key in agg:
            agg[key] += counts[key]

    report = {
        "dataset": config.ws.dataset,
        "mode": config.ws.mode,
        "selector": SELECTOR,
        "selection_schema_version": hio.SELECTION_SCHEMA_VERSION,
        "items": len(items),
        "k": config.k,
        "status_counts": status_counts,
        "mean_honest_p_htrue": (sum(p_htrue_selected) / len(p_htrue_selected)
                                if p_htrue_selected else None),
        "gates": agg,
        "fallback_items": fallback_items,
        "verifier_model": config.model,
        "option_seed": config.option_seed,
        "score_prompt_version": score_verifier.PROMPT_VERSION,
        "run_manifest_digest": bundle["digest"],
        "dry_run": config.dry_run,
    }
    return selections, report


def run_select_stage(ws, config, warn=hio.stderr_warn):
    bundle = verify_pool(ws, config, "select", warn)
    selections, report = process(bundle, config)
    ws.ensure_dirs()
    hio.atomic_write_jsonl(ws.selected, selections)
    hio.atomic_write_json(ws.report, report)
    counts = report["status_counts"]
    print("[honest-select/select] items=%d k=%d ok=%d fallback=%d none=%d "
          "mean_p_htrue=%s" % (report["items"], config.k, counts[STATUS_OK],
                               counts[STATUS_FALLBACK], counts[STATUS_NONE],
                               report["mean_honest_p_htrue"]))
    print("[honest-select/select] -> %s  report -> %s" % (ws.selected, ws.report))
    return 0


# ---- write stage ----------------------------------------------------------
def _qy_core(q_y):
    return {k: q_y.get(k) for k in ("question", "Y_true", "Y_false")}


def check_dataset_alignment(items, canonical, items_path, canonical_path):
    """The three dataset files must be same-length, same-order, same-content."""
    if len(items) != len(canonical):
        raise hio.HonestBokError("length mismatch: %s has %d rows, %s has %d"
                                 % (items_path, len(items), canonical_path, len(canonical)))
    for index, (item, canon) in enumerate(zip(items, canonical)):
        if item.get("story_title") != canon.get("story_title"):
            raise hio.HonestBokError("story_title mismatch at index %d" % index)
        if _qy_core(item.get("Q_Y") or {}) != _qy_core(canon.get("Q_Y") or {}):
            raise hio.HonestBokError("Q_Y mismatch at index %d" % index)
        if (item.get("Q_H") or {}) != (canon.get("Q_H") or {}):
            raise hio.HonestBokError("Q_H mismatch at index %d" % index)


def check_selection_row(row, item, index, bundle, config):
    """Re-verify one selection row against the pool before publishing it."""
    errors = hio.validate_honest_selection(row, k=config.k)
    if errors:
        raise hio.HonestBokError("write: selection row %d is invalid: %s"
                                 % (index, errors[0]))
    if row.get("dataset_index") != index:
        raise hio.HonestBokError("write: selection row %d has dataset_index %r"
                                 % (index, row.get("dataset_index")))
    if row.get("selector") != SELECTOR:
        raise hio.HonestBokError("write: selection row %d used selector %r"
                                 % (index, row.get("selector")))
    selected = row.get("honest_selected")
    if selected is None:
        raise hio.HonestBokError(
            "write: item %d has no selected honest transcript (status=%r). With a "
            "complete pool the fallback always selects, so this indicates a bug."
            % (index, row.get("status")))
    if selected.get("story_title") != item.get("story_title"):
        raise hio.HonestBokError("write: selection story_title mismatch at index %d" % index)
    if _qy_core(selected.get("qy") or {}) != _qy_core(item.get("Q_Y") or {}):
        raise hio.HonestBokError("write: selection Q_Y mismatch at index %d" % index)
    selected_qh, item_qh = selected.get("qh") or {}, item.get("Q_H") or {}
    for key in ("question", "H_true", "H_false"):
        if selected_qh.get(key) != item_qh.get(key):
            raise hio.HonestBokError("write: selection Q_H.%s mismatch at index %d"
                                     % (key, index))

    # The embedded candidate/score must be byte-identical to the verified pool rows.
    key = hio.candidate_key(selected)
    pool_candidate = bundle["candidates_by_key"].get(key)
    if pool_candidate is None:
        raise hio.HonestBokError("write: selected candidate %r is not in the pool" % (key,))
    embedded = {k: v for k, v in selected.items() if k != "score"}
    if embedded != pool_candidate:
        raise hio.HonestBokError(
            "write: selected candidate %r differs from the pool row; selected.jsonl is "
            "stale relative to candidate.jsonl" % (key,))
    if selected.get("score") != bundle["scores_by_key"].get(key):
        raise hio.HonestBokError(
            "write: selected score %r differs from verifier_scores.jsonl" % (key,))

    transcript = selected["transcript"]
    if len(transcript["rounds"]) != config.num_rounds:
        raise hio.HonestBokError("write: index %d transcript has %d round(s), expected %d"
                                 % (index, len(transcript["rounds"]), config.num_rounds))
    return transcript


def build_publication(bundle, selections, config):
    """Validate everything, then return (new_canonical, new_with_honest, stats).

    Nothing is written here: a half-written canonical dataset is the one failure mode
    that would be expensive to recover from.
    """
    ws = config.ws
    items = bundle["items"]
    canonical = hio.load_dataset_items(ws.canonical_source)
    check_dataset_alignment(items, canonical, ws.items, ws.canonical_source)

    by_index = {}
    for position, row in enumerate(selections):
        index = row.get("dataset_index")
        if not isinstance(index, int) or not 0 <= index < len(items):
            raise hio.HonestBokError("write: selection row %d has out-of-range index %r"
                                     % (position, index))
        if index in by_index:
            raise hio.HonestBokError("write: duplicate selection for dataset index %d" % index)
        by_index[index] = row
    if len(by_index) != len(items):
        missing = [i for i in range(len(items)) if i not in by_index]
        raise hio.HonestBokError(
            "write: selected.jsonl covers %d/%d items (missing: %s)"
            % (len(by_index), len(items), missing[:10]))

    transcripts = {}
    for index, item in enumerate(items):
        transcripts[index] = check_selection_row(by_index[index], item, index,
                                                 bundle, config)

    new_canonical = copy.deepcopy(canonical)
    for index, row in enumerate(new_canonical):
        # In-place assignment on the existing key preserves q_m key order, and every
        # other field (notably q_m.adversarial_transcript) is left untouched.
        row["Q_Y"]["honest_transcript"] = transcripts[index]

    new_with_honest = copy.deepcopy(items)
    for index, row in enumerate(new_with_honest):
        row["Q_Y"]["transcript"] = transcripts[index]

    stats = {
        "items": len(items),
        "honest_transcript": len(transcripts),
        "adversarial_transcript": sum(
            1 for row in new_canonical if row["Q_Y"].get("adversarial_transcript") is not None),
    }
    return canonical, new_canonical, new_with_honest, stats


def make_publication_verifier(ws, original_canonical, items, transcripts_expected):
    """Post-stage / post-commit check: only q_m.honest_transcript may have changed."""

    def verify(reloaded):
        canonical = reloaded[ws.canonical_out]
        with_honest = reloaded[ws.with_honest_out]
        if len(canonical) != len(original_canonical):
            raise hio.HonestBokError("publication: canonical length changed")
        if len(with_honest) != len(items):
            raise hio.HonestBokError("publication: with-honest length mismatch")
        filled = 0
        for index, (before, after) in enumerate(zip(original_canonical, canonical)):
            if after["Q_Y"].get("honest_transcript") is not None:
                filled += 1
            if before["Q_Y"].get("adversarial_transcript") != after["Q_Y"].get("adversarial_transcript"):
                raise hio.HonestBokError("publication: adversarial_transcript changed at %d" % index)
            if _qy_core(before["Q_Y"]) != _qy_core(after["Q_Y"]):
                raise hio.HonestBokError("publication: Q_Y changed at %d" % index)
            if before.get("Q_H") != after.get("Q_H"):
                raise hio.HonestBokError("publication: Q_H changed at %d" % index)
            if before.get("story_title") != after.get("story_title"):
                raise hio.HonestBokError("publication: story_title changed at %d" % index)
            if list(before["Q_Y"].keys()) != list(after["Q_Y"].keys()):
                raise hio.HonestBokError("publication: Q_Y key order changed at %d" % index)
        if filled != transcripts_expected:
            raise hio.HonestBokError("publication: honest_transcript count %d != %d"
                                     % (filled, transcripts_expected))
        for index, row in enumerate(with_honest):
            if sorted(row["Q_Y"].keys()) != ["Y_false", "Y_true", "question", "transcript"]:
                raise hio.HonestBokError("publication: with-honest Q_Y keys wrong at %d" % index)
            if row["Q_Y"]["transcript"] != canonical[index]["Q_Y"]["honest_transcript"]:
                raise hio.HonestBokError("publication: the two files disagree at %d" % index)

    return verify


def run_write_stage(ws, config, warn=hio.stderr_warn):
    recovered = hio.recover_publication(ws, warn)
    if recovered:
        print("[honest-select/write] recovered an interrupted publication: %s" % recovered)

    bundle = verify_pool(ws, config, "write", warn)
    selections = hio.read_jsonl_strict(ws.selected, "selections")

    # Re-run the selector from the verified bundle and require ``selected.jsonl`` to
    # equal the deterministic H-BoK-Htrue result. Checking only that each selected row
    # is internally well-formed and present in the pool is not enough: any other pool
    # candidate is equally well-formed, so a hand-edited or stale selected.jsonl could
    # publish a non-argmax (or gate-excluded) transcript. Selection is deterministic, so
    # equality is the right assertion.
    expected, _ = process(bundle, config)
    if selections != expected:
        differing = [row.get("dataset_index")
                     for row, want in zip(selections, expected) if row != want]
        raise hio.HonestBokError(
            "write: %s does not match the recomputed H-BoK-Htrue result (%d row(s) differ, first "
            "at index %s; %d rows on disk vs %d recomputed). It is stale or was edited; "
            "re-run --stage select."
            % (ws.selected, len(differing), differing[:5] or None,
               len(selections), len(expected)))

    original_canonical, new_canonical, new_with_honest, stats = build_publication(
        bundle, selections, config)

    print("[honest-select/write] mode=%s items=%d honest_transcript=%d "
          "adversarial_transcript(preserved)=%d"
          % (ws.mode, stats["items"], stats["honest_transcript"],
             stats["adversarial_transcript"]))

    verify = make_publication_verifier(ws, original_canonical, bundle["items"],
                                       stats["honest_transcript"])
    written = hio.publish_transaction(
        ws,
        [(ws.with_honest_out, new_with_honest), (ws.canonical_out, new_canonical)],
        verify, warn)
    print("[honest-select/write] committed %s" % (written,))
    return 0


# ---- CLI ------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score and select honest best-of-K candidates (H-BoK-Htrue).",
        epilog="There are no path flags: every input and output is derived from "
               "--dataset. --dry-run confines the whole run (including publication) to "
               "runs/honest/<DS>/dryrun/ and can never write the canonical dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=hio.DATASETS)
    p.add_argument("--stage", choices=["all", "score", "select", "write"], default="all")
    p.add_argument("--k", type=hio.positive_int, default=hio.DEFAULT_K,
                   help="must match the generation manifest; completeness is enforced")
    p.add_argument("--num-rounds", type=hio.positive_int, default=hio.DEFAULT_NUM_ROUNDS)
    p.add_argument("--model", default=hio.DEFAULT_VERIFIER_MODEL,
                   help="UNFINE-TUNED base verifier checkpoint")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--option-seed", type=int, default=score_verifier.DEFAULT_OPTION_SEED,
                   help="must match the adversarial run so both arms share one A/B order")
    p.add_argument("--overwrite", action="store_true", help="re-score from scratch")
    p.add_argument("--rehash-verifier", action="store_true",
                   help="ignore the fingerprint cache and re-hash the checkpoint in full, "
                        "both when scoring binds the fingerprint and when select/write "
                        "re-verify it")
    p.add_argument("--dry-run", action="store_true",
                   help="sandbox rehearsal in runs/honest/<DS>/dryrun/ with fake scores")
    return p.parse_args(argv)


def main(argv=None, workspace=None):
    """CLI entry point. `workspace` is a TEST-ONLY internal injection point."""
    args = parse_args(argv)
    ws = workspace or hio.Workspace(
        args.dataset, mode=hio.MODE_DRYRUN if args.dry_run else hio.MODE_PRODUCTION)
    config = SelectConfig(ws, k=args.k, num_rounds=args.num_rounds, model=args.model,
                          device=args.device, option_seed=args.option_seed,
                          dry_run=args.dry_run, rehash_verifier=args.rehash_verifier)
    try:
        hio.assert_production_layout(ws)
        if args.stage in ("all", "score"):
            run_score_stage(ws, config, overwrite=args.overwrite)
        if args.stage in ("all", "select"):
            run_select_stage(ws, config)
        if args.stage in ("all", "write"):
            run_write_stage(ws, config)
    except hio.HonestBokError as exc:
        print("[honest-select] ABORT: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
