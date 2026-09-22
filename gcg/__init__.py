"""GCG (Greedy Coordinate Gradient) tools for the verifier experiments.

It reuses ``adversarial_transcript``'s prompt builders, candidate schema and atomic
IO so the optimized readout stays identical to the production one, but it operates
on a frozen causal LM and a versioned task specification and never changes the
production candidate/selection path by itself.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
