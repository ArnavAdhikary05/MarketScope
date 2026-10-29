# Testing and verification

## Deterministic backend tests

Run from the repository root:

```powershell
Set-Location .\backend
python -m unittest discover -s tests -v
```

The model-comparison tests generate synthetic OHLCV locally and verify chronological fold counts, a selected horizon forecast, manual naive selection, and invalid-model rejection. They do not contact Yahoo Finance.

`backend\test_provider.py` is an optional manual provider smoke check and requires network/provider availability. It is not part of the deterministic suite.

## Frontend checks

Run from the repository root:

```powershell
Set-Location .\frontend
npm run build
npm run lint
```

## Current coverage gaps

There are no automated API integration tests, browser/component tests, database persistence tests, Prophet-specific tests, out-of-sample final-test protocol, or end-to-end Docker validation. Build/lint/test outcomes should be recorded from the current environment; no model performance or trading return is asserted by this documentation.
