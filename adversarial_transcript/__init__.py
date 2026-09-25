"""Adversarial debate candidate generation and selection.

For each paired item, the release pipeline generates ``K=8`` public ``q_m`` debates while
the debater defending ``Y_true`` privately targets ``H_false`` on ``q_h``. The package
then scores candidates with the frozen base verifier and applies the fail-closed
Adv-BoK-Hfalse selector.

Offline schemas, filters, and selection remain dependency-light and consume external
score JSONL. Generation imports the shared debate engine lazily. Stored transcripts retain
the native stance map and round structure; condition and selection metadata remain outside
the public transcript object.
"""

__all__ = [
    "common",
    "schema",
    "filters",
    "select",
    "score",
    "score_verifier",
    "generate",
    "run_pipeline",
    "prompt",
    "fingerprint",
    "readout",
]
__version__ = "0.1.0"
