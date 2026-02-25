# Agent and collaborator reference

## Primary source files

- `kv-cache-tradeoffs.md`: target architecture and scaffolding.
- `Memory — GitHub Repo.md`: paper framing and experiment intent.

## Key decisions

- Source-of-truth experiments: `configs/` only.
- Raw results kept separate from processed outputs.
- Every figure must have a paired config and generation command.
- Current benchmark path is synthetic by default (torch.randint input) with real KV cache behavior.

## Contact points

- Primary maintainer notes are in `guidelines.md`.
- For execution questions, check the nearest TODO in `feature-queue.md`.
