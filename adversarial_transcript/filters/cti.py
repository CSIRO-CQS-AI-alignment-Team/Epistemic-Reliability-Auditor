"""CTI ranking - select by MAXIMUM p_hfalse.

After the hard gates, survivors are ranked by the base verifier's transcript-
conditioned hidden readout, p_hfalse = P_Vbase(H_false | Q_H, T). In the default
pipeline this selects T_adv only. When the honest-BoK control is explicitly
enabled, the same ranking is applied to both arms.

Tie-break: highest p_hfalse, then lowest candidate_idx (deterministic).
"""


def rank(scored_survivors):
    """Pick the argmax-p_hfalse candidate from [(candidate, score), ...].

    Returns (selected_candidate, selected_score) or (None, None) if empty.
    """
    if not scored_survivors:
        return None, None
    # max p_hfalse; on ties prefer the smaller candidate_idx (so -idx maximised).
    best = max(
        scored_survivors,
        key=lambda cs: (float(cs[1]["p_hfalse"]), -int(cs[0]["candidate_idx"])),
    )
    return best[0], best[1]


def readout(adv_score, honest_score=None):
    """Build the CTI readout dict from selected scores (honest may be absent)."""
    adv_p = float(adv_score["p_hfalse"]) if adv_score else None
    hon_p = float(honest_score["p_hfalse"]) if honest_score else None
    delta = (adv_p - hon_p) if (adv_p is not None and hon_p is not None) else None
    src = adv_score or honest_score or {}
    return {
        "adv_p_hfalse": adv_p,
        "honest_bok_p_hfalse": hon_p,
        "delta": delta,
        "baseline_p_hfalse": src.get("baseline_p_hfalse"),
        "scorer": src.get("scorer"),
    }
