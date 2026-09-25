"""Validate the score record's q_m no-regression diagnostic.

The diagnostic passes when the transcript does not overturn a visible question that
the base verifier answered correctly without the transcript:

    pass iff qy_clean_correct is False OR qy_adv_correct is True

The production selector applies the stricter transcript-conditioned `q_m=Y_true`
constraint directly. This helper remains useful for validating the accompanying score
provenance. It is fail-closed when the score or required boolean fields are missing.
"""


def check(candidate, score):
    """Return (passed, info). `score` is the joined external score dict or None."""
    if score is None:
        return False, {"reason": "no_score"}

    clean = score.get("qy_clean_correct")
    adv = score.get("qy_adv_correct", score.get("qy_correct"))
    preserved = score.get("qy_no_regression_pass")
    if clean is None:
        return False, {"reason": "qy_clean_correct_missing"}
    if adv is None:
        return False, {"reason": "qy_adv_correct_missing"}
    if preserved is None:
        return False, {"reason": "qy_no_regression_pass_missing"}
    if not isinstance(clean, bool):
        return False, {"reason": "qy_clean_correct_not_boolean"}
    if not isinstance(adv, bool):
        return False, {"reason": "qy_adv_correct_not_boolean"}
    if not isinstance(preserved, bool):
        return False, {"reason": "qy_no_regression_pass_not_boolean"}
    formula = (not clean) or adv
    if preserved is not formula:
        return False, {"reason": "qy_no_regression_mismatch", **_info(score, qy_preserved=formula)}
    if formula is True:
        return True, _info(score, qy_preserved=True)
    if formula is False:
        return False, _info(score, qy_preserved=False)
    return False, {"reason": "qy_no_regression_unreachable"}


def _info(score, qy_preserved):
    return {
        "gate": "no_regression",
        "qy_clean_correct": score.get("qy_clean_correct"),
        "qy_adv_correct": score.get("qy_adv_correct", score.get("qy_correct")),
        "qy_no_regression_pass": qy_preserved,
        "qy_clean_pred_letter": score.get("qy_clean_pred_letter"),
        "qy_adv_pred_letter": score.get("qy_adv_pred_letter", score.get("qy_pred_letter")),
    }
