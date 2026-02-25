# Contributor and agent guidelines

## Collaboration protocol

- Read and update status in `project-status.md` before major changes.
- Move items only in one direction (`planned -> in_progress -> done`).
- Include references to config files in every feature entry.
- Prefer deterministic code paths and seed control.
- Use one PR/task per hypothesis per benchmark type.

## Coding standards

- Type hints required for all public interfaces.
- Reproducibility by default: seeded RNG, logged hyperparameters, explicit config paths.
- No silent CLI defaults that alter experiment semantics.
- Artifacts are versioned if produced for figures or claims.
- CLI JSON output should include runtime metadata (`python`, `torch`, `cuda`, `gpu`) for comparability.
- Benchmark quality claims should be explicitly marked synthetic unless trained checkpoints are used.
