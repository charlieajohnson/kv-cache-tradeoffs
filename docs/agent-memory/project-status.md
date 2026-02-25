# Project status

## Current state

- Stage: Initial scaffolding complete
- Focus: Deterministic benchmark core and experiment execution path

## Risk log

- No model checkpoints in repo by design (not versioned).
- GPU-specific memory metrics need hardware-specific calibration.
- Statistical reporting not yet automated end-to-end.

## Next actions

1. Implement full attention modules with exact shape assertions.
2. Wire benchmark loops to emit canonical CSV/JSONL logs.
3. Add CI gating for CLI smoke + static checks.
