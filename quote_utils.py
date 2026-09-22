"""Dependency-light text normalization and debate quote verification.

The judge, transcript selectors, and verifier-training pipeline reuse these stdlib-only
helpers without importing model clients or GPU dependencies.
"""

import re
import string


def normalize_text(text):
    """Normalize for quote matching (mirrors parser.normalize_text): curly->straight
    quotes, strip all punctuation, lowercase, collapse whitespace."""
    text = text.replace("”", '"').replace("“", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    return " ".join(text.split())


def verify_quotes(argument, story_normalised):
    """Re-tag each <quote> as <v_quote> (verified) or <u_quote> (unverified) by an exact
    normalized-substring match against the story (mirrors parser.verify_strict). The
    quote's displayed text is left unchanged; only the surrounding tag changes.
    """
    # Collapse any pre-existing verified/unverified tags back to plain <quote> first.
    for tag in ("<v_quote>", "<u_quote>"):
        argument = argument.replace(tag, "<quote>")
    for tag in ("</v_quote>", "</u_quote>"):
        argument = argument.replace(tag, "</quote>")

    def change_tag(match):
        quote = match.group(1)
        if normalize_text(quote) in story_normalised:
            return f"<v_quote>{quote}</v_quote>"
        return f"<u_quote>{quote}</u_quote>"

    return re.sub(r"<quote>(.*?)</quote>", change_tag, argument)
