# Testing and verification

## Deterministic backend tests

Run from the repository root:

```powershell
Set-Location .\backend
python -m unittest discover -s tests -v
```

Forecast comparison tests use generated local OHLCV fixtures to verify chronological validation, numeric-horizon backwards compatibility, named-horizon exchange-session dates, empirical interval output, manual baseline selection, and invalid or unsupported requests. Sentiment tests mock Groq and provider responses to cover missing credentials, malformed classifications, provider fallback, timestamp alignment, correlation edge cases, ticker validation, and 30-day filtering. They do not contact live news, Groq, or Yahoo Finance.

`backend\test_provider.py` is an optional manual provider smoke check and requires network/provider availability. It is not part of the deterministic suite.

## Frontend checks

Run from the repository root:

```powershell
Set-Location .\frontend
npm run build
npm run lint
```

## Current coverage gaps

There are no browser/component tests, database persistence tests, LSTM/PyTorch-specific tests, final untouched test protocol, or end-to-end Docker validation. API route functions have deterministic unit coverage, but no HTTP-client integration suite is configured. Build/lint/test outcomes should be recorded from the current environment; no model performance or trading return is asserted by this documentation.
