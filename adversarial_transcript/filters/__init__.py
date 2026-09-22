"""Selection filters for the adversarial_transcript funnel (all stdlib-only).

Each filter exposes `check(...) -> (passed: bool, info: dict)` and never mutates
its inputs. The safety-gate ORDER is fixed (GATES); cti ranking runs AFTER the gates, over
the safe candidates that the absolute Q_Y constraint admits (the primary branch). select.py
composes these; keeping them as pure functions makes each independently testable offline.
"""

from adversarial_transcript.filters import correctness, leakage, quote_audit, cti

# Hard safety gates, applied in this order. `score` checks that the joined external
# readout is schema-valid and usable; the absolute Q_Y=Y_true requirement is applied
# afterward as the primary-branch constraint. A candidate that fails that branch
# constraint may still enter the Q_Y fallback. `correctness` is an independent
# no-regression diagnostic and is not part of GATES.
GATES = ("score", "leakage", "quote_audit")

__all__ = ["correctness", "leakage", "quote_audit", "cti", "GATES"]
