# Shared-template mask engine

This module implements the production mask pipeline used by the UAV CLI and Python API.

## Pipeline

Stage 1 discovers candidates on the first frame, every eight frames and the final frame. It aggregates fragments, records original-image evidence, and tracks independent instances against shared templates. Predictions never count as new image evidence.

Stage 2 evaluates template relationships and trajectory continuity using retained observations. It refines active template groups, reuses accepted SAM results, and performs one bounded feedback pass before planning final masks. Template identity and instance lifetime remain separate.

Final masks project shared templates through each instance pose, apply the configured dilation and uncertainty policy, and pass directly to ProPainter with the original decoded frames in memory. Debug output is optional.

## Code organization

- stage2.py orchestrates association, refinement and final output planning.
- association.py and reconciliation.py separate evidence evaluation from reversible relationship updates.
- ownership.py resolves pixels around independently moving objects.
- output_plan.py and output_geometry.py share the final occurrence and geometry decisions used by masks and review rendering.
- motion.py, gaps.py and edge_trajectory.py handle measured trajectories, missing intervals and frame-clipped observations.
- rendering.py produces final masks; review modules create optional visualizations.

Evidence and derived caches are scoped to a run and kept in RAM. Explicit Stage 1 checkpoints are an internal debugging facility, never automatically read or written by the production CLI. Load only trusted checkpoints.

See the root README for supported parameters, installation, output layouts and third-party license conditions.
