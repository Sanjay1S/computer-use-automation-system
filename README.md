# understudy

A computer-use automation system. An LLM discovers how to complete a task in a web application **once**. The run is compiled into a typed, versioned, reviewable **capability**. Agents then invoke that capability through **deterministic replay**, with no model in the loop. Replay reports business outcomes, recoverable conditions and hard failures as distinct results, and a human operator can take over the live browser session when automation can't safely continue.

```
goal ─► discovery (LLM, sandbox only) ─► draft capability ─► review + approve (by content hash)
                                                                      │
        caller ─► replay (no LLM): preflight → steps → outcome race → typed result ◄─┘
                         │ stuck or risky
                         └─► operator takes over the same live session, then hands it back
```

The design write-up is [REPORT.md](REPORT.md). A worked end-to-end example, with logs from a real discovery run and from replays that succeed, hit a business outcome, recover, fail and hand off, is in [evidence/](evidence/README.md).

The demo target is [ParaBank](https://parabank.parasoft.com) (a banking sandbox with server-rendered pages and no test IDs). The capability is **Open New Account**: inputs `account_type` and `funding_account_id`, output `new_account_id`.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker, only for the local ParaBank used by the demo (`parasoft/parabank`)
- An `ANTHROPIC_API_KEY`, only for **discovery**. Replay never calls a model.

## Setup

```bash
make setup            # uv sync, then install Playwright's Chromium
cp .env.example .env  # then fill it in; .env is gitignored
```

`.env` holds `ANTHROPIC_API_KEY` (discovery only), `PARABANK_USERNAME` / `PARABANK_PASSWORD` (ParaBank's published demo login is `john` / `demo`; they go only to the login step, never to the model, logs or artifacts) and `PARABANK_PORT` (host port for the local container, default 8080; the `parabank-local` tenant URL follows it). `make` reads `.env` itself. For `uv run` commands, load it first: `set -a; . ./.env; set +a`.

## Run without live services

```bash
make check   # ruff, ruff format --check, mypy (strict, all of src/)
make test    # 567 tests, about two minutes: no network, API key or Docker
```

The suite drives a routed fake origin (`https://fixture.test`) that copies ParaBank's real DOM, plus a scripted fake model that exercises the whole chain (discover, compile, replay) offline. Tests marked `live` (`make test-live`) need the container, credentials and, for one of them, the API key; they are skipped by default.

## Demo path

```bash
make parabank-up                 # local ParaBank on $PARABANK_PORT
set -a; . ./.env; set +a

# 1. Discovery: the model drives a real browser until the goal is met, then the run is compiled.
#    It stops at the commit and waits for you: open the console URL it prints and click Approve.
uv run understudy discover goals/open_account.yaml --tenant parabank-local
#    -> capabilities/drafts/parabank.open_account.json, plus a run folder under runs/

# 2. Review the draft: contract, steps, locator fallback chains with their reasoning, lint result.
uv run understudy review  capabilities/drafts/parabank.open_account.json
uv run understudy approve capabilities/drafts/parabank.open_account.json --reviewer "<your name>"
#    -> capabilities/parabank.open_account@1.0.0.json, bound to the content hash

# 3. Replay, no model. The commit needs an approval for this invocation: --approve.
uv run understudy replay parabank.open_account --tenant parabank-local --approve \
    --input account_type=CHECKING --input funding_account_id=13344           # succeeded + new_account_id

# 4. The agent-facing contract (a tool definition a calling agent can read).
uv run understudy describe parabank.open_account --as tool
```

The repo already contains an approved artifact and a finished discovery run, so you can start at step 3 (the committed artifact was discovered on the public ParaBank instance and replays here unchanged). `13344` is one of the demo customer's accounts on a fresh container; account ids differ per instance. Discovery can also run on `parabank-public`, Parasoft's shared demo site; because it is shared, we kept it to a single discovery run (our rule, not enforced in code). Prefer the local container.

### Error and exceptional states

```bash
R="uv run understudy replay parabank.open_account --tenant parabank-local --approve"
$R --input account_type=SAVINGS --input funding_account_id=99999                          # business_outcome (exit 4)
$R --input account_type=SAVINGS --input funding_account_id=13344 --inject session_expired_at=s3   # recovered
$R --input account_type=SAVINGS --input funding_account_id=13344 --inject rename_target:s4        # failed: target_not_found (exit 1)
$R --input account_type=SAVINGS --input funding_account_id=13344 --inject drop_commit_response    # needs_human: commit_outcome_unknown (exit 5)
uv run understudy replay parabank.open_account --tenant parabank-local \
    --input account_type=CHECKING --input funding_account_id=13344                        # rejected: approval_required (exit 3)
```

Each command prints the result as JSON. Exit codes: 0 succeeded, 1 failed, 2 usage, 3 rejected, 4 business outcome, 5 needs a human. Each run writes `runs/<run_id>/`: redacted events, masked screenshots, a failure bundle (masked screenshot, sanitized DOM, accessibility snapshot, console) when something went wrong, and `result.json`. `--inject` works on sandbox tenants only.

## Operator handoff

With `--on-stuck escalate`, a run that reaches a state it cannot handle safely (an unknown overlay, a commit whose answer never came) waits for a person instead of returning `needs_human`. With `--approval interactive`, the commit waits for an operator's approval. Both print the address of the operator console (`127.0.0.1:8765` with a random token; `--console-port` to change it).

```bash
uv run understudy replay parabank.open_account --tenant parabank-local --approve --headed \
    --on-stuck escalate --inject unknown_modal_at:s3 \
    --input account_type=SAVINGS --input funding_account_id=13344
```

1. Open the console URL in a normal browser and **Claim**. Input you give before claiming is not recorded as yours.
2. In the **browser window the run opened**, close the notice. It is the same session the automation was using.
3. Back in the console, **Resume**. Replay re-checks the paused step and carries on. (**Complete** asks replay to verify the success state and read the outputs, which it refuses for a commit that hasn't been sent; **Abort** ends the run.)

The console is a deliberately minimal, mock operator surface: a view-only screenshot that refreshes every 2 s, with the controls. You act in the browser window, not in the picture, so a takeover needs `--headed` and a display. Don't open the console URL inside the automation window: its policy guard refuses it. On a remote machine, forward the console port (`ssh -L 8765:127.0.0.1:8765 …`) to open the console, and run the headed window where a display exists. Every handoff is recorded in the run folder (`interventions/`, `handoff.json`, events with `actor: human`). `evidence/06b_replay_handoff_headed/` is a real example.

## Layout

| Path | What it holds |
|---|---|
| `src/understudy/artifact/` | the capability schema, signature DSL, lint rules (L001–L011), content hashing, review and tool-definition export |
| `src/understudy/surface/` | the `Surface` seam and its Playwright implementation: snapshots and refs, locator synthesis and resolution, masked capture, request guard |
| `src/understudy/discovery/` | the browser-toolset driver, recorder tools, agent loop, stuck detection, and the compiler from trajectory to draft |
| `src/understudy/replay/` | preflight, the outcome race, recovery, the result contract, and the engine |
| `src/understudy/policy/`, `redaction/` | the single policy gate (hosts, routes, actions, commit rules) and the redactor |
| `src/understudy/control/` | the control plane (lease and fencing token), interventions, the console, the human-action recorder |
| `apps/parabank/`, `tenants/`, `goals/` | the vendor app profile and policy, tenant bindings, and the discovery goal |
| `capabilities/` | the approved artifact (and drafts) |
| `evidence/` | curated runs: discovery, replays, handoffs, and the approved artifact |
