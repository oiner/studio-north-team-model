# Reproducibility and runtime evidence

These checks were run on 2026-11-18 on a Windows laptop CPU with Python 3.12.14 and uv 0.12.7.

1. **Clean copy.** The entry-point files (`main.py`, `models.py`, `history.py`, `pyproject.toml`, `uv.lock`, `prediction_grid.csv`, `submission-contract.json`) were copied to an empty folder, then `uv sync --frozen` was run.
2. **Two runs.** Each used the standard command against the eight released CSVs (dataset 5.1) with seed 20261116:

   ```text
   uv run --frozen python main.py --data-dir INPUTS --prediction-grid prediction_grid.csv --output outputs/runN/predictions.csv --seed 20261116
   ```

   - Both runs printed `{"status": "validated", "rows": 1152}`.
   - Model time was about 8.5 s, and wall time about 13 s per run, including Python start-up.
3. **Prediction bytes are identical.** Both runs produced SHA-256 `7a0ec04abed95717db1b240a21fa2d270fb42884f479b7b39ba49a8b63e9366e`. Receipts are in [`evidence/`](evidence/). They record the source, `uv.lock`, contract, grid and input hashes, plus `future_outcomes_read: false`.
4. **The instructor's public validator passed.** The run was `uv run --frozen python lab.py --output outputs/student validate-submission --csv <run1>/predictions.csv` in the lab repository at commit `0e40ce3cd0b788a3914ca4eb9f90d58ae1b62903`. It returned `status: passed` for 1,152 rows.
5. **Manual inspection** of the raw model output found no clipping or adjustment anywhere in `predict`:

   | Check | Result |
   |---|---|
   | Grid coverage | All 1,152 published keys: 96 cells × 12 months, with exposures unchanged |
   | Units: retained | Expected members, min 3.9; never above 0.987 × origin survivors |
   | Units: net revenue | USD; $56–$89 per expected member-month |
   | Units: contribution | USD; $26–$44 per expected member-month, 46–51% of net revenue |
   | Membership | Strictly decreasing in every cell; largest month-to-month change is −0.057 |
   | Nonnegativity | All values nonnegative and finite |

Passing the format checks says nothing about accuracy on the private holdout.
