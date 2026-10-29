# Market data sources

The current provider adapter uses `yfinance` to retrieve OHLCV history. It is intended for educational research and does not guarantee real-time data, exchange entitlements, complete coverage, or production service levels. Provider limits and retention rules may change.

The adapter accepts Yahoo Finance symbols such as `AAPL` and `RELIANCE.NS` when the provider supports them. It normalizes column names, removes missing or structurally invalid OHLCV rows, sorts observations, and removes duplicate timestamps. It uses `auto_adjust=True`; the returned prices are adjusted according to yfinance behavior and should not be confused with unadjusted exchange prints.

Supported provider interval values include 1m, 5m, 15m, 30m, 60m, 1h, and 1d, subject to Yahoo's currently available history. The UI exposes daily, 5-minute, and 15-minute choices. Data freshness is reported as the timestamp of the last returned observation; this does not establish that the feed is live.

The platform currently does not use a licensed provider, exchange calendar, market-status endpoint, retry policy, rate-limit coordinator, corporate-action store, or persistent bar cache. Daily future dates are approximated with weekdays; intraday timestamps are arithmetic offsets and may fall outside trading sessions.
