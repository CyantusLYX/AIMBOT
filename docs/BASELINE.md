# AIMBOT — Refactor Baseline (Phase 0)

This file records the current refactor baseline so future runs can be compared
consistently.

## Snapshot

- Date: 2026-03-14
- Branch: `refactor/architecture-v1`
- Commit (HEAD at baseline capture): `c80a76b`
- Python: `3.10.11`

## Reference Inputs

- Primary smoke video: `data/DJI_20250422132606_0030_D.MP4`
- Alternative videos:
  - `data/DJI_20250422132836_0032_D.MP4`
  - `data/DJI_20250422133233_0034_D.MP4`
  - `data/DJI_20250422133441_0035_D.MP4`
- YOLO weights: `models/epoch_149.pt`

## Baseline Validation Commands

```powershell
# Syntax check
uv run python -m compileall src scripts

# End-to-end dry-run smoke
uv run python scripts/run_pipeline.py --dry-run --source data/DJI_20250422132606_0030_D.MP4
```

## Current Baseline Output

- Validation output: terminal logs from `compileall` and `run_pipeline.py`
- Current status: `passed` (latest run)

## Target Metrics To Track In Future Iterations

- Tracking continuity (ID switches / target lock stability)
- Reacquisition success after occlusion
- Effective FPS (average + P95 frame latency)
- Gimbal command output stability and frequency

## Notes

- This baseline captures post-refactor validation entry points.
- Strict pre-refactor vs post-refactor equivalence still requires preserving a
  comparable pre-refactor metrics artifact for side-by-side comparison.
