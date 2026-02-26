# Proposal Assets

This directory contains proposal-ready artifacts generated from evaluation outputs.

## Regenerate assets

Run the command below from the repository root:

```bash
python scripts/make_proposal_assets.py --eval outputs/eval_small --out docs/proposal_assets
```

## Expected input files

- `outputs/eval_small/leaderboard.csv` (required)
- `outputs/eval_small/calibration.png` (optional; copied if present)

## Generated output files

- `docs/proposal_assets/results_table.md`
- `docs/proposal_assets/README.md`
- `docs/proposal_assets/calibration.png` (optional)
