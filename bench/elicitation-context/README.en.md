# Elicitation context comparison

This opt-in check measures gateway usage without changing the number, order or schemas of
expert questions. It is not a production LLM plugin and does not start a model estimate.

Set `GATEWAY_BASE_URL`, `GATEWAY_API_KEY`, `ELICIT_BENCH_ROOT` (a new output directory),
`ELICIT_BENCH_SOURCE` (default `workspace`) and optionally `ELICIT_BENCH_MODEL`
(default `gpt-6-astra`). Run:

```powershell
.\.venv\Scripts\python.exe bench/elicitation-context/check.py
```

The script copies series to isolated workspaces, elicits one law and eight response questions
through the actual MCP tool. The second mode reuses the elicited law unchanged, so differences
are not caused by a different size distribution. The executor supplies no numeric answers.
Only the eight matching response questions enter the token comparison; the shared law is
reported separately in the trace.

- `history`: retain all preceding tool turns.
- `bounded`: reset messages for every question; supply only the current self-contained envelope.
- `per-variable`: retain the four separate horizon questions for one variable, then reset.

Every question still asks nine response numbers. Context includes the saved law, earlier
horizons of the same variable and one available reference response. Resetting context does not
make the economic answer correct: signs, conditioning, units, persistence and uncertainty must
be examined separately. Three matching size nodes are not three independent shocks.

Interrupted checks resume from saved parts and per-mode message checkpoints. Transient gateway
429/502/503/504 and timeouts get at most three attempts, with a bounded backoff. A timeout may
have consumed tokens without reporting usage; those costs are explicitly unknown, not zero.
Do not run modes concurrently against the same gateway limit.

Traces record gateway `prompt_tokens`, `completion_tokens` and reported cached tokens. They
contain the supplied numerical answers but no credentials. Token reduction is not monetary
savings: billing weights, caching and model prices can differ. Repeated independent trials
are needed before claiming a stable quality effect.

The normal MCP interview server cannot erase the caller's DSH history. To realize the measured
context reduction, the caller must use fresh narrow contexts at the tested boundary. A normal
coding subagent with the parent's whole tool catalog may not reproduce this saving.
