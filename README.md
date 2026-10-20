# understudy

An LLM discovers how to complete a task in a web application once. The run is compiled into a typed, versioned, reviewable **capability**. Agents then invoke that capability through **deterministic replay**, with no model in the loop. Replay reports business outcomes, recoveries and failures as distinct results, and a human operator can take over the live browser session when automation can't safely continue.

Work in progress. Setup, the demo path and the design write-up (`REPORT.md`) land as the build proceeds.

## Development

```bash
make setup   # install dependencies and Playwright's Chromium
make check   # lint, format check, type check
make test    # offline test suite: no network, API keys or Docker needed
```

## Replay against local ParaBank

```bash
make parabank-up                      # docker run parasoft/parabank (PARABANK_PORT, default 8080)
cp .env.example .env                  # set PARABANK_USERNAME / PARABANK_PASSWORD (demo: john / demo)
D=capabilities/drafts/parabank.open_account@1.0.0.json
uv run understudy replay $D --tenant parabank-local --allow-draft --approve \
    --input account_type=SAVINGS --input funding_account_id=12345        # succeeded
uv run understudy replay $D --tenant parabank-local --allow-draft --approve \
    --input account_type=SAVINGS --input funding_account_id=99999        # business outcome
uv run understudy replay $D ... --inject session_expired_at=s3           # recovered
uv run understudy replay $D ... --inject rename_target:s4                # failed: target_not_found
uv run understudy replay $D ... --inject drop_commit_response            # needs_human
make test-live                        # the same flow as a test (needs the container and credentials)
```

The command prints the result as JSON and exits 0 (succeeded), 1 (failed), 3 (rejected), 4 (business outcome) or 5 (needs a human). Each run writes `runs/<run_id>/`: redacted events, masked screenshots, a failure bundle when something went wrong, and `result.json`.
