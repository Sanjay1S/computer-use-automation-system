# understudy

An LLM discovers how to complete a task in a web application once. The run is compiled into a typed, versioned, reviewable **capability**. Agents then invoke that capability through **deterministic replay**, with no model in the loop. Replay reports business outcomes, recoveries and failures as distinct results, and a human operator can take over the live browser session when automation can't safely continue.

Work in progress. Setup, the demo path and the design write-up (`REPORT.md`) land as the build proceeds.

## Development

```bash
make setup   # install dependencies and Playwright's Chromium
make check   # lint, format check, type check
make test    # offline test suite: no network, API keys or Docker needed
```
