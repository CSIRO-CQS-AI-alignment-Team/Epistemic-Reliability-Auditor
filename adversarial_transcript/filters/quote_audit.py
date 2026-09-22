"""Quote audit (hard) — debaters' <quote> spans must be grounded in the story.

Reuses quote_utils.verify_quotes (the SAME verifier the debaters/judge saw): each
<quote>...</quote> is re-tagged <v_quote> (exact normalized substring of the story)
or <u_quote> (not found). We count verified vs. unverified across every round/both
debaters and drop a candidate whose unverified FRACTION exceeds the threshold. The
default threshold used by select.py is 0.0, so any unverified quote fails when story
text is available; transcripts with no quotes trivially pass.

If no story text is available, the default production behavior is to fail closed.
Callers can explicitly set require_story=False for smoke/debug runs that need to
skip quote grounding.
"""

from quote_utils import normalize_text, verify_quotes

from adversarial_transcript.common import iter_round_args


def audit(candidate, story):
    """Return quote counts for a candidate against `story` (raw story text)."""
    story_norm = normalize_text(story)
    verified = unverified = 0
    for _name, arg in iter_round_args(candidate["transcript"]):
        tagged = verify_quotes(arg, story_norm)
        verified += tagged.count("<v_quote>")
        unverified += tagged.count("<u_quote>")
    total = verified + unverified
    frac = (unverified / total) if total else 0.0
    return {"n_quotes": total, "n_verified": verified, "n_unverified": unverified,
            "unverified_frac": frac}


def check(candidate, story, max_unverified_frac=0.0, require_story=True):
    """Return (passed, info). `story` is the raw story text, or None to SKIP.

    Passes if there are no quotes, or the unverified fraction is within threshold.
    """
    if story is None:
        return (not require_story), {
            "skipped": not require_story,
            "reason": "no_story_text",
            "require_story": require_story,
        }
    info = audit(candidate, story)
    info["skipped"] = False
    info["max_unverified_frac"] = max_unverified_frac
    passed = info["n_quotes"] == 0 or info["unverified_frac"] <= max_unverified_frac
    return passed, info
