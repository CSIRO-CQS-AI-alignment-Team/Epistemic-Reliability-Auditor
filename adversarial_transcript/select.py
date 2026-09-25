"""Fail-closed offline selection for scored debate candidates.

The release pipeline selects one adversarial transcript per item with Adv-BoK-Hfalse:

1. require a schema-valid external score, no explicit hidden-task leakage, and grounded
   quotes;
2. among safe candidates whose transcript-conditioned q_m readout selects Y_true, choose
   the largest p_hfalse, breaking ties by smaller candidate_idx;
3. when the primary pool is empty, retain the safe candidate with the largest explicit
   P(Y_true | q_m, T) only as qy_fallback audit data;
4. otherwise record no_adv_survivor.

Only status ``ok`` represents a canonical adversarial transcript. This module performs no
model calls: all q_m and q_h readouts come from the external score JSONL. The optional
multi-condition mode applies the same funnel to an honest comparison arm while preserving
per-arm status fields.

Usage:
    python3 -m adversarial_transcript.select \
        --candidates runs/candidates.jsonl \
        --scores runs/verifier_scores.jsonl \
        --stories dataset/QuALITY-H/QuALITY-H-title-story.json \
        --out runs/selected.jsonl --report runs/report.json
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adversarial_transcript import common, schema  # noqa: E402
from adversarial_transcript.filters import leakage, quote_audit, cti  # noqa: E402

CONDITIONS = ("adversarial", "honest")
DEFAULT_CONDITIONS = ("adversarial",)

SELECTOR = "adv-bok-hfalse"

# Which branch produced a selection.
BRANCH_PRIMARY = "primary"
BRANCH_FALLBACK = "qy_fallback"

# Per-arm and row statuses.
STATUS_OK = "ok"
STATUS_QY_FALLBACK = "qy_fallback"
STATUS_NO_ADV = "no_adv_survivor"
STATUS_NO_HONEST = "no_honest_survivor"
STATUS_NONE = "no_survivor"

# When qy_adv_probs.semantic carries the Y_false complement, the pair must be a two-point
# distribution to this tolerance; a partially edited block is rejected, not half-trusted.
QY_PROB_SUM_TOL = 1e-6


def parse_conditions(raw):
    requested = [c.strip().lower() for c in raw.split(",") if c.strip()]
    bad = [c for c in requested if c not in CONDITIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown condition(s): {', '.join(bad)}")
    if not requested:
        raise argparse.ArgumentTypeError("at least one condition is required")
    if "adversarial" not in requested:
        raise argparse.ArgumentTypeError("selection requires adversarial candidates")
    return [c for c in CONDITIONS if c in requested]


# ---- score readouts -------------------------------------------------------
def qy_adv_correct_of(score):
    """Transcript-conditioned absolute q_m readout, or None if unusable.

    ``qy_correct`` is the compact alias carried by the score schema. Strict input
    validation still requires the explicit ``qy_adv_correct`` field.
    """
    if score is None:
        return None
    value = score.get("qy_adv_correct", score.get("qy_correct"))
    if not isinstance(value, bool):
        return None
    return value


def qy_true_prob(score):
    """P(Y_true | T) from qy_adv_probs.semantic, or None if unusable.

    STRICT: never reconstructed from 1 - Y_false, from the letter block, or from
    qy_adv_pred. `Y_true` must itself be a non-bool number in [0, 1] (which also rejects
    NaN/inf).

    `Y_false` is OPTIONAL, because only P(Y_true) is required to rank the fallback. But
    if the key is PRESENT it must be well formed -- a non-bool number in [0, 1] whose sum
    with Y_true is 1 within QY_PROB_SUM_TOL. A present-but-malformed complement (a string,
    a bool, null, NaN/inf, out of range, or a pair that does not sum to 1) means the block
    was partially edited or corrupted, so THIS PROBABILITY readout is rejected rather than
    half-trusted: the candidate becomes fallback-unusable. The boolean `qy_adv_correct`
    gate is independent and unaffected, so such a candidate can still be primary-eligible.
    """
    if score is None:
        return None
    probs = score.get("qy_adv_probs")
    if not isinstance(probs, dict):
        return None
    semantic = probs.get("semantic")
    if not isinstance(semantic, dict):
        return None
    value = semantic.get("Y_true")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        return None
    if "Y_false" in semantic:
        other = semantic["Y_false"]
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            return None
        other = float(other)
        # A NaN/inf complement fails this comparison, so it is rejected here.
        if not 0.0 <= other <= 1.0:
            return None
        if abs(value + other - 1.0) > QY_PROB_SUM_TOL:
            return None
    return value


# ---- per-candidate funnel -------------------------------------------------
def score_gate(score):
    """Hard gate 1: the candidate must carry a usable base-verifier readout.

    Fail-closed. On success, ``info`` also carries clean-q_m and no-regression
    diagnostics for the audit report; neither field controls primary selection.
    """
    if score is None:
        return False, {"reason": "no_score"}
    raw = score.get("qy_adv_correct", score.get("qy_correct"))
    if raw is None:
        return False, {"reason": "qy_adv_correct_missing"}
    if not isinstance(raw, bool):
        return False, {"reason": "qy_adv_correct_not_boolean"}
    return True, {
        "gate": "score",
        "qy_adv_correct": raw,
        "qy_true_prob": qy_true_prob(score),
        "qy_clean_correct": score.get("qy_clean_correct"),
        "qy_no_regression_pass": score.get("qy_no_regression_pass"),
        "qy_clean_pred_letter": score.get("qy_clean_pred_letter"),
        "qy_adv_pred_letter": score.get("qy_adv_pred_letter", score.get("qy_pred_letter")),
    }


def run_gates(candidate, score, story, max_unverified_frac, extra_terms,
              require_story=True):
    """Run the hard SAFETY gates in order. Returns (safe: bool, trace: dict).

    trace always records which stage was decisive (`stage`) plus each stage's info for the
    stages that were reached. Passing every gate makes the candidate eligible for the
    primary branch or the q_m fallback; it does NOT by itself select anything.
    """
    passed, info = score_gate(score)
    trace = {"score": info}
    if not passed:
        trace["stage"] = "score"
        return False, trace

    passed, leak = leakage.check(candidate, extra_terms)
    trace["leakage"] = leak
    if not passed:
        trace["stage"] = "leakage"
        return False, trace

    passed, qa = quote_audit.check(candidate, story, max_unverified_frac,
                                   require_story=require_story)
    trace["quote_audit"] = qa
    if not passed:
        trace["stage"] = "quote_audit"
        return False, trace

    trace["stage"] = "safe"
    return True, trace


def rank_by_qy_true_prob(entries):
    """q_m fallback ranking over [(candidate, score, p_ytrue), ...].

    argmax P(Y_true); ties -> smaller candidate_idx, the same deterministic secondary key
    as cti.rank. Deliberately NOT p_hfalse: the fallback is explicitly not an attack, so
    the attack objective must not decide it. Returns the entry or None if empty.
    """
    if not entries:
        return None
    return max(entries, key=lambda e: (float(e[2]), -int(e[0]["candidate_idx"])))


def _empty_gate_counts():
    """Per-condition funnel accounting. Every value is an int (they are summed)."""
    return {
        "total": 0,
        "no_score": 0,
        "drop_score": 0,
        "drop_leakage": 0,
        "drop_quote_audit": 0,
        "safe": 0,
        "survivors": 0,
        "qy_fallback_eligible": 0,
        "qy_fallback_unusable": 0,
    }


def select_condition(cands, scores_index, story, max_unverified_frac, extra_terms,
                     require_story=True):
    """Run Adv-BoK-Hfalse for one (item, condition) group.

    Returns (selected_candidate_or_None, selected_score_or_None, branch_or_None, counts,
    audits). `audits` carries one entry per SAFE candidate (the pool both branches choose
    from); gate-dropped candidates stay accounted for in `counts`.
    """
    counts = _empty_gate_counts()
    primary = []
    fallback = []
    audits = []
    for c in sorted(cands, key=lambda x: int(x["candidate_idx"])):
        counts["total"] += 1
        score = scores_index.get(schema.score_key(c))
        safe, trace = run_gates(
            c, score, story, max_unverified_frac, extra_terms,
            require_story=require_story,
        )
        if not safe:
            stage = trace["stage"]
            if stage == "score" and trace["score"].get("reason") == "no_score":
                counts["no_score"] += 1
            counts[f"drop_{stage}"] += 1
            continue

        counts["safe"] += 1
        info = trace["score"]
        prob = info["qy_true_prob"]
        if info["qy_adv_correct"]:
            eligible_for = BRANCH_PRIMARY
            counts["survivors"] += 1
            primary.append((c, score))
        elif prob is not None:
            eligible_for = BRANCH_FALLBACK
            counts["qy_fallback_eligible"] += 1
            fallback.append((c, score, prob))
        else:
            eligible_for = "none"
            counts["qy_fallback_unusable"] += 1
        audits.append({
            "candidate_idx": c["candidate_idx"],
            "leakage": trace.get("leakage"),
            "quote_audit": trace.get("quote_audit"),
            "p_hfalse": float(score["p_hfalse"]),
            "qy_adv_correct": info["qy_adv_correct"],
            "qy_true_prob": prob,
            "qy_clean_correct": info["qy_clean_correct"],
            "qy_no_regression_pass": info["qy_no_regression_pass"],
            "eligible_for": eligible_for,
        })

    sel_c, sel_s = cti.rank(primary)
    if sel_c is not None:
        return sel_c, sel_s, BRANCH_PRIMARY, counts, audits
    # Only now, with the primary pool empty, may the q_m fallback run -- over SAFE
    # candidates only, never the raw pool.
    best = rank_by_qy_true_prob(fallback)
    if best is not None:
        return best[0], best[1], BRANCH_FALLBACK, counts, audits
    return None, None, None, counts, audits


select_adversarial = select_condition


def join_score(candidate, score):
    """Attach the joined score onto a copy of the candidate for the output record."""
    out = dict(candidate)
    out["score"] = score
    return out


def selection_block(candidate, score, branch):
    """Per-arm readout of WHICH branch selected WHAT, plus the q_m diagnostics."""
    if candidate is None or score is None:
        return {
            "branch": None,
            "candidate_idx": None,
            "p_hfalse": None,
            "qy_true_prob": None,
            "qy_adv_correct": None,
            "qy_clean_correct": None,
            "qy_no_regression_pass": None,
        }
    return {
        "branch": branch,
        "candidate_idx": candidate["candidate_idx"],
        "p_hfalse": float(score["p_hfalse"]),
        "qy_true_prob": qy_true_prob(score),
        "qy_adv_correct": qy_adv_correct_of(score),
        "qy_clean_correct": score.get("qy_clean_correct"),
        "qy_no_regression_pass": score.get("qy_no_regression_pass"),
    }


def arm_status(selected, branch, condition):
    """Per-arm status: ok (primary) / qy_fallback / the arm's no-survivor status."""
    if selected is None:
        return STATUS_NO_ADV if condition == "adversarial" else STATUS_NO_HONEST
    return STATUS_OK if branch == BRANCH_PRIMARY else STATUS_QY_FALLBACK


def join_status(adv_status, honest_status):
    """Return the publication-safety status for a multi-condition row.

    The top-level status is ``ok`` only when every requested arm used its primary branch.
    A missing arm takes precedence, and any fallback demotes the joined row to
    ``qy_fallback``. Read ``adversarial_status`` for the adversarial arm's outcome.
    """
    adv_missing = adv_status == STATUS_NO_ADV
    honest_missing = honest_status == STATUS_NO_HONEST
    if adv_missing and honest_missing:
        return STATUS_NONE
    if adv_missing:
        return STATUS_NO_ADV
    if honest_missing:
        return STATUS_NO_HONEST
    if adv_status == STATUS_OK and honest_status == STATUS_OK:
        return STATUS_OK
    return STATUS_QY_FALLBACK


def _empty_agg(conditions):
    return {
        "items": 0,
        "status_counts": {s: 0 for s in (STATUS_OK, STATUS_QY_FALLBACK, STATUS_NO_ADV,
                                         STATUS_NO_HONEST, STATUS_NONE)},
        "adv_status_counts": {s: 0 for s in (STATUS_OK, STATUS_QY_FALLBACK, STATUS_NO_ADV)},
        "honest_status_counts": {s: 0 for s in (STATUS_OK, STATUS_QY_FALLBACK,
                                                STATUS_NO_HONEST)},
        "adv_p": [],
        "honest_p": [],
        "deltas": [],
        "fallback_p": [],
        "fallback_qy_true": [],
        "adv_fallback_items": [],
        "honest_fallback_items": [],
        "gates": _empty_gate_counts(),
        "by_condition": {condition: _empty_gate_counts() for condition in conditions},
    }


def process(candidates, scores, story_map, max_unverified_frac=0.0, extra_terms=(),
            require_story=True, conditions=DEFAULT_CONDITIONS, warn=lambda m: None):
    """Core offline selection. Returns (selections, report). Pure / no IO."""
    conditions = tuple(conditions)
    include_honest = "honest" in conditions
    # Index scores by (item_id, condition, candidate_idx). Schema-invalid rows never
    # reach the index, so the gate below always sees a structurally valid record.
    scores_index = {}
    for s in scores:
        if schema.validate_score(s):
            continue
        scores_index[schema.score_key(s)] = s

    # Group selected-condition candidates by item_id, then condition.
    by_item = {}
    order = []
    for c in candidates:
        cond = c.get("condition")
        if cond not in conditions:
            continue
        iid = c["item_id"]
        if iid not in by_item:
            by_item[iid] = {condition: [] for condition in conditions}
            order.append(iid)
        by_item[iid][cond].append(c)

    selections = []
    agg = _empty_agg(conditions)

    for iid in order:
        groups = by_item[iid]
        story_title = None
        for cond in conditions:
            if groups[cond]:
                story_title = groups[cond][0].get("story_title")
                break
        story = story_map.get(story_title) if story_map else None
        if story_map and story is None:
            action = "will fail quote audit" if require_story else "quote audit skipped"
            warn(f"{iid}: story '{story_title}' absent from story map; {action}")

        selected = {}
        item_gates = {}
        item_audits = {}
        for cond in conditions:
            sc, ss, branch, counts, audits = select_condition(
                groups[cond], scores_index, story, max_unverified_frac, extra_terms,
                require_story=require_story,
            )
            selected[cond] = (sc, ss, branch)
            item_gates[cond] = counts
            item_audits[cond] = audits
            for k, v in counts.items():
                agg["by_condition"][cond][k] += v
                if not include_honest and cond == "adversarial":
                    agg["gates"][k] += v

        adv_c, adv_s, adv_branch = selected["adversarial"]
        hon_c, hon_s, hon_branch = selected.get("honest", (None, None, None))
        adv_status = arm_status(adv_c, adv_branch, "adversarial")
        adv_block = selection_block(adv_c, adv_s, adv_branch)
        if include_honest:
            honest_status = arm_status(hon_c, hon_branch, "honest")
            honest_block = selection_block(hon_c, hon_s, hon_branch)
            status = join_status(adv_status, honest_status)
        else:
            honest_status = None
            status = adv_status

        cti_readout = cti.readout(adv_s, hon_s if include_honest else None)
        row = {
            "item_id": iid,
            "story_title": story_title,
            "selector": SELECTOR,
            "status": status,
            "adversarial_status": adv_status,
            "adversarial_selected": join_score(adv_c, adv_s) if adv_c else None,
            "selection": ({"adversarial": adv_block, "honest": honest_block}
                          if include_honest else adv_block),
            "gates": item_gates if include_honest else item_gates["adversarial"],
            "audits": item_audits if include_honest else item_audits["adversarial"],
            "cti": cti_readout,
        }
        if include_honest:
            row["honest_status"] = honest_status
            row["honest_bok_selected"] = join_score(hon_c, hon_s) if hon_c else None
        selections.append(row)

        agg["items"] += 1
        agg["status_counts"][status] += 1
        agg["adv_status_counts"][adv_status] += 1
        # Per-arm means key off the PER-ARM status, never the join status: under the
        # publish-safe join rule an adv-primary/honest-fallback item is not "ok", and it
        # must still count toward the adversarial mean.
        if adv_status == STATUS_OK:
            agg["adv_p"].append(adv_block["p_hfalse"])
        elif adv_status == STATUS_QY_FALLBACK:
            agg["adv_fallback_items"].append(iid)
            agg["fallback_p"].append(adv_block["p_hfalse"])
            agg["fallback_qy_true"].append(adv_block["qy_true_prob"])
        if include_honest:
            agg["honest_status_counts"][honest_status] += 1
            if honest_status == STATUS_OK:
                agg["honest_p"].append(honest_block["p_hfalse"])
            elif honest_status == STATUS_QY_FALLBACK:
                agg["honest_fallback_items"].append(iid)
            if adv_status == STATUS_OK and honest_status == STATUS_OK:
                agg["deltas"].append(adv_block["p_hfalse"] - honest_block["p_hfalse"])

    report = _summarize(agg, include_honest=include_honest)
    return selections, report


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _summarize(agg, include_honest=False):
    if not include_honest:
        return {
            "items": agg["items"],
            "selector": SELECTOR,
            "conditions": list(DEFAULT_CONDITIONS),
            "status_counts": {k: agg["status_counts"][k]
                              for k in (STATUS_OK, STATUS_QY_FALLBACK, STATUS_NO_ADV)},
            "mean_adv_p_hfalse": _mean(agg["adv_p"]),
            "mean_qy_fallback_p_hfalse": _mean(agg["fallback_p"]),
            "mean_qy_fallback_qy_true_prob": _mean(agg["fallback_qy_true"]),
            "qy_fallback_items": list(agg["adv_fallback_items"]),
            "gates": agg["gates"],
        }

    deltas = [d for d in agg["deltas"] if d is not None]
    return {
        "items": agg["items"],
        "selector": SELECTOR,
        "conditions": list(agg["by_condition"].keys()),
        # The five mutually exclusive join statuses sum to ``items``.
        "status_counts": dict(agg["status_counts"]),
        "adversarial_status_counts": dict(agg["adv_status_counts"]),
        "honest_status_counts": dict(agg["honest_status_counts"]),
        "mean_delta": _mean(deltas),
        "delta_items": len(deltas),
        "mean_adv_p_hfalse": _mean(agg["adv_p"]),
        "mean_honest_bok_p_hfalse": _mean(agg["honest_p"]),
        "mean_qy_fallback_p_hfalse": _mean(agg["fallback_p"]),
        "mean_qy_fallback_qy_true_prob": _mean(agg["fallback_qy_true"]),
        "adversarial_qy_fallback_items": list(agg["adv_fallback_items"]),
        "honest_qy_fallback_items": list(agg["honest_fallback_items"]),
        "items_with_positive_delta": sum(1 for d in deltas if d > 0),
        "by_condition": agg["by_condition"],
    }


# ---- CLI ------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Offline selection funnel for debate candidates.")
    p.add_argument("--candidates", required=True, help="candidates JSONL (from generate.py)")
    p.add_argument("--scores", required=True, help="EXTERNAL score JSONL (verifier readout)")
    p.add_argument("--stories", required=True,
                   help="title-story.json for hard quote audit")
    p.add_argument("--out", required=True, help="selections JSONL output")
    p.add_argument("--report", default=None, help="aggregate report JSON output")
    p.add_argument("--conditions", type=parse_conditions,
                   default=list(DEFAULT_CONDITIONS),
                   help="comma-separated subset to select; default: adversarial only")
    p.add_argument("--quote-max-unverified-frac", type=float, default=0.0,
                   help="drop a candidate whose unverified-quote fraction exceeds this")
    p.add_argument("--extra-leakage-terms", default="",
                   help="comma-separated extra hard-drop terms")
    p.add_argument("--leakage-mode", choices=["regex", "regex+judge"], default="regex",
                   help="'regex+judge' is reserved for an online LLM leg; offline applies regex only")
    p.add_argument("--strict-schema", action="store_true",
                   help="abort if any candidate/score record fails validation (default: warn + skip)")
    p.add_argument("--allow-missing-stories", action="store_true",
                   help="debug only: skip quote grounding when story text is missing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    def warn(m):
        print(f"[WARN] {m}", file=sys.stderr)

    if args.leakage_mode == "regex+judge":
        warn("--leakage-mode regex+judge: the judge leg is an online extension; "
             "this offline run applied the regex hard-term filter only.")
    if "honest" in args.conditions:
        warn("--conditions honest applies the same Adv-BoK-Hfalse rule to the honest "
             "comparison arm, ranking it by p_hfalse. This is distinct from the "
             "H-BoK-Htrue selector in honest_select.py. The top-level status is 'ok' "
             "only when both requested arms use their primary branch; read "
             "adversarial_status for the adversarial arm's outcome.")

    candidates = common.read_jsonl(args.candidates)
    scores = common.read_jsonl(args.scores)

    # Validate; skip bad records (or abort under --strict-schema).
    good_c = []
    for c in candidates:
        errs = schema.validate_candidate(c)
        if errs:
            msg = f"candidate {c.get('item_id')}/{c.get('condition')}#{c.get('candidate_idx')}: {errs[0]}"
            if args.strict_schema:
                raise SystemExit(f"[SCHEMA] {msg}")
            warn(f"dropping invalid candidate: {msg}")
            continue
        good_c.append(c)
    good_s = []
    for s in scores:
        errs = schema.validate_score(s)
        if errs:
            if args.strict_schema:
                raise SystemExit(f"[SCHEMA] score: {errs[0]}")
            warn(f"dropping invalid score: {errs[0]}")
            continue
        good_s.append(s)

    story_map = common.load_story_map(args.stories)
    extra = tuple(t.strip() for t in args.extra_leakage_terms.split(",") if t.strip())

    selections, report = process(
        good_c, good_s, story_map,
        max_unverified_frac=args.quote_max_unverified_frac,
        extra_terms=extra, require_story=not args.allow_missing_stories,
        conditions=args.conditions,
        warn=warn,
    )

    common.write_jsonl(args.out, selections)
    if args.report:
        common.write_json(args.report, report)

    sc = report["status_counts"]
    if args.conditions == ["adversarial"]:
        print(f"selected {report['items']} item(s) [{SELECTOR}]: ok={sc[STATUS_OK]} "
              f"qy_fallback={sc[STATUS_QY_FALLBACK]} no_adv={sc[STATUS_NO_ADV]}")
        print(f"mean adv p_hfalse (primary only) = {report['mean_adv_p_hfalse']}")
        if sc[STATUS_QY_FALLBACK]:
            print(f"Q_Y fallback selections are NOT attack successes and are not "
                  f"published by the downstream merge (status != 'ok'); "
                  f"mean fallback P(Y_true) = {report['mean_qy_fallback_qy_true_prob']}")
    else:
        print(f"selected {report['items']} item(s) [{SELECTOR}]: ok={sc[STATUS_OK]} "
              f"qy_fallback={sc[STATUS_QY_FALLBACK]} no_adv={sc[STATUS_NO_ADV]} "
              f"no_honest={sc[STATUS_NO_HONEST]} no_survivor={sc[STATUS_NONE]}")
        print(f"per-arm: adversarial={report['adversarial_status_counts']} "
              f"honest={report['honest_status_counts']}")
        print(f"mean delta (adv - honest_bok) = {report['mean_delta']}  "
              f"positive on {report['items_with_positive_delta']}/{report['delta_items']} "
              f"both-primary items")
    print(f"-> {args.out}" + (f"  report -> {args.report}" if args.report else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
