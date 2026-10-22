# Design write-up

The model discovers; the artifact becomes a reusable capability; deterministic replay is how an agent invokes it in production. The slice runs against ParaBank (server-rendered, AJAX panels hidden until needed, `;jsessionid` in URLs, no test IDs) with one flow, *Open New Account*. Evidence for each claim is in [`evidence/`](evidence/README.md).

## 1. Architecture

```
 DISCOVERY (LLM in loop, sandbox only)                       REPLAY (no LLM, production path)
 goal ─► agent loop ─► browser-toolset driver ─┐             capability@version + inputs + tenant
 recorder ─► trajectory ─► compiler ─► draft ──┼─► review + approve (by hash) ─► ReplayEngine
              one PolicyGate ◄─────────────────┴──── same policy file ───────► one PolicyGate
              one Surface seam (Playwright): observe · resolve · act · probe · capture
              one control plane (lease + fencing token) ◄──► operator console
```

- **Two paths, split at the artifact.** The model runs only in discovery, on a sandbox; replay never calls it, so production is deterministic, cheap, and keeps regulated data from any model. Replay was built first, so discovery had to emit what a proven consumer accepts.
- **Mechanism.** Anthropic's native browser toolset, executed by our own Playwright driver. Grounding is the accessibility tree (frames included), a screenshot on request, and coordinates only as a fallback hit-tested back to an element, which works without a clean DOM.
- **Own loop, not the SDK runner:** every step is policy-gated, recorded with the model's stated reason, budgeted (40 turns, 10 min, $5), pausable for a person, and testable with a scripted fake model.
- **The `Surface` seam.** Neither path touches Playwright directly. A new surface is a new implementation plus new locator kinds; the artifact doesn't change.
- **Rejected:** a custom tool schema (the trained action space comes free); pixel control for the web (doesn't replay robustly); agent frameworks (they hide the loop and the locator decisions); LLM self-healing in production (see Cuts).
- **Simple:** one process, sync Python, Pydantic v2, canonical JSON so approval binds to a hash. 584 offline tests, mypy strict.

## 2. Artifact schema

A calling agent reads `description`, `inputs`, `outputs`, `side_effects` and `outcomes`, never `steps` (`understudy describe --as tool` renders a tool definition); a reviewer reads the same file with `understudy review`.

```jsonc
{ "id": "parabank.open_account", "version": "1.0.0",
  "lifecycle": { "status": "approved", "approved_hash": "sha256:c7114ccc…" },
  "inputs":  [ { "name": "funding_account_id", "type": "string", "pattern": "^\\d+$", "sensitivity": "financial_id" } ],
  "outputs": [ { "name": "new_account_id", "source_step": "s5_read_new_account_id" } ],
  "side_effects": { "class": "commit", "approval": "per_invocation" },
  "outcomes": [ { "code": "funding_account_id_not_found", "detected_by": "s3_select_funding_account_id" } ],
  "steps": [ { "id": "s4_click_open_new_account", "action": "click", "effect": "commit",
      "target": { "fingerprint": { "role": "button", "name": "Open New Account" },
                  "locators": [ { "by": "role", "stability": "high" }, { "by": "css", "stability": "medium" }, … ] },
      "expect": [ { "when": { "kind": "visible", "role": "heading", "name": "Account Opened!" }, "then": { "do": "continue" } } ],
      "retry": {} } ],
  "success": { "signature": …, "negative_control": … } }
```

- **Contract first.** Inputs are typed (enum, pattern, sensitivity). Secrets are references, the base URL is a tenant binding, and no session id is baked in.
- **Targets carry a verified fallback chain plus a fingerprint.** Each candidate resolved to exactly one element, *the same one*, at discovery, and carries a stability tag and a note saying why. A fallback must match the fingerprint, so replay never clicks a look-alike.
- **Every step has its own checkpoint** (`expect`) in a small signature DSL (`visible`, `absent`, `url`, `selected`, `options_loaded`, `dialog`, `http_status`; `all`/`any`/`not`). The success checkpoint has a **negative control**: it must be false on the start state, because ParaBank's result panels sit hidden in the DOM from load and "text is present" would pass before anything happened.
- **Effect class** (`read` < `input` < `commit`) drives safety. Approval binds to the **content hash**; any edit is a new draft.
- **Lint L001–L012** runs in the compiler, before approval and at load: vacuous success, sensitive literals, a retried commit, unbound params, unverified targets, policy drift, outputs without extractors, baked-in base URLs or session ids, and (at compile time, when the run's values are known) any of the run's inputs or outputs in free text. Contract prose is written from the checkpoints, never copied from the model.

## 3. Determinism & error handling

- **No sleeps, no model.** Before acting: Playwright auto-wait plus the step's `wait_before`. After acting, an **outcome race** polls every 100 ms in priority order: the app's global states (session expired, app error), the step's `expect` list, then dialogs. Network idle is never a success signal (legacy apps poll). Viewport, locale and timezone are fixed.
- **Resolution.** Locators are tried in order; the first giving one visible element *with a matching fingerprint* wins (a fallback win emits `drift`). If all fail, `target_not_found` lists every attempt. Every action is hit-tested first: Playwright checks interception only for clicks, and would otherwise fill a form hidden behind a modal.

| Condition | Detected by | Result |
|---|---|---|
| Bad input, draft artifact, commit without approval | preflight, before the UI is touched | `rejected` |
| Funding account not the customer's | options loaded, value absent | `business_outcome: funding_account_id_not_found` |
| Slow list; 5xx or timeout before the commit | `wait_before`; status or race timeout | normal; recovery `retry_transient` |
| Session expired before the commit | login heading appears in place (no redirect) | recovery `reauthenticate`, restart from re-entry point |
| Known interstitial | app-profile signature | recovery `dismiss_interstitial` |
| Unknown overlay or dialog | hit-test or dialog | `needs_human: unknown_blocking_state` |
| No answer, or an app error, after the commit | race timeout or global state | `needs_human: commit_outcome_unknown`, `commit_state=unknown` |
| Button renamed or removed | all locators fail or fingerprint mismatch | `failed: automation/target_not_found`, with attempts and a failure bundle |
| Off-allowlist navigation; crash or DNS failure | request guard; Playwright error | `failed: policy/…`; `failed: infra/…` (apart from app failures) |

- **Principle:** after a commit, anything short of a definite success or a definite business rejection is `commit_state=unknown` and goes to a person, **never a retry**, because money may have moved. Recoveries are a finite registry with a budget of 3, all reported.
- **Runtime negative control:** before the commit, replay checks the success checkpoint is false on the live page, or fails without sending anything.
- **Fault injection** (`--inject`, sandbox only) covers each row; the scenario matrix asserts status, code, commit state and what reached the app. **UI drift is secondary**, surfacing as `drift` events.

## 4. Heterogeneity & multi-tenant

*Designed, not built, apart from what is noted.*

- **Surfaces.** The seam is `Surface` plus the `Target` fingerprint. **Legacy web** adds frame paths, table-relative locators (column header plus row key) and label-proximity locators (already used here). **Desktop** adds a UI Automation surface (ControlType→role, Name→name, AutomationId→test id); discovery uses the computer-use toolset and hit-tests the point to an element. Citrix/RDP is vision-only (OCR anchors), the lowest stability tier, always with canaries and review.
- **Reuse.** Three layers: a shared **vendor app profile** (signatures, handlers, policy defaults, redaction); a vendor-scoped **capability** with `compatible_versions`; a **tenant binding** (base URL, secret refs, enabled capabilities, policy *tightening*). Per-tenant differences are overrides keyed by step and locator, each separately approved, never loosening policy. Shown cheaply: the artifact discovered on the public ParaBank replays unchanged on the local one (evidence `02`), and a real label difference between them ("Account Overview" vs "Accounts Overview") is kept out of checkpoints.
- **Drift.** Passive: drift events aggregated per capability × tenant × app version. Active: canaries (commits as a dry run up to the step before). On drift the capability is degraded for that tenant and re-discovery yields an override diff **for human review**; production is never patched automatically.

## 5. Escalation & handoff

- **Stuck, in discovery:** the model calls `request_human`; the same action on one target three times with no state change; three consecutive errors; no URL or DOM change in 5 actions; 80% of a budget spent without success; a refusal. **In replay:** an unknown blocking state, a commit with an unknown outcome, an approval required.
- **The request** carries the capability and hash, the step and intent, the reason, the sanitized URL, a masked live screenshot, recent events and the allowed resolutions.
- **Control transfer: a lease with a fencing token.** `AGENT` → `AWAITING_HUMAN` (paused) → `HUMAN` (claimed) → `AGENT` on resume. Every actuation goes through `Surface.act(…, lease)`; a stale holder raises `ControlLost`, so a paused thread can never act after a handover. Only the claimer can resume or complete.
- **Same live session.** The person uses the headed browser window the automation was driving. The console is a deliberate mock (view-only screenshot plus controls); a remote screencast relay is design only (seam: `Control.relay`).
- **What the person did is recorded:** an injected recorder reports trusted click, change and submit events with redacted values, attributed against the lease log. Input outside `HUMAN` is an `unexpected_input` warning. In discovery a person's steps compile into the draft, flagged for review.
- **Resume** re-checks the paused step. **Complete** makes replay verify success and read the outputs (refused for an unsent commit). **Abort** fails the run. Commits stay blocked at the network while a person holds the session, and earlier inputs are re-verified before the commit.
- **Evidence:** `06` (headless: decide only) and `06b` (headed: the person's click logged as `actor: human`, resume at the paused step, verified success).

## 6. Safety

- **One `PolicyGate`** from a per-app policy a tenant can only tighten: hosts, a route **allowlist** (exactly what the in-scope flow was seen to request: its pages, assets and two XHRs) plus a denylist (`/admin.htm`, `/initializeDB.htm` and others), browser actions and keys, commit rules. Every refused request is logged as a `policy` event, so a missing route is visible, not silent. Enforced at five points: the model's tool configuration, the SDK `url_policy` and `confirm` hooks, a browser-level route guard on every request, a per-step check in replay, and lint at load.
- **Risky actions require confirmation, bound to the exact invocation.** Blocking outright would stop the system reaching the confirmation screen; flagging lets money move before anyone looks. It mirrors maker-checker. Risk is judged on the **resolved element**, not the browser action (a purchase is just a click), and the commit route stays blocked at the network except inside a single-use window.
- **Redaction.** Field-aware classification first, regex backstop. Secrets are never written; PII becomes a keyed HMAC tag; financial ids keep at most a third of their digits; URLs lose `jsessionid`; screenshots are masked at capture, DOM snapshots sanitized, traces never enabled. Credentials reach only the login step. A scan of `evidence/` for the password, session ids, API key, customer name and account ids found zero hits.
- **Model data.** Discovery runs on sandbox tenants only. Injection defences are layered: page text framed as untrusted, classifiers on tool results, every action through the gate, a human on every commit and artifact.
- **Limits.** Redaction is heuristic: novel PII formats can slip through, and any 5-digit run in free text is masked as an id. A same-host redirect to another port or scheme is detected only after it is sent. Routes are allowlisted per app, not yet per capability: a second ParaBank capability would widen what the first may touch. The console is localhost with a random token and a self-declared operator; approval tokens are a CLI flag standing in for a signed grant. L012 catches only values the run actually used; a value that merely looks like customer data is left to redaction.

## 7. Cuts

- **Left out, deliberately:** multi-tenant overlay resolution and desktop and legacy-web surfaces (designed above, seams in place); a real-time console and screencast relay (control transfer is real, the surface a mock); queues, services, a scheduler, egress isolation, a secrets vault and a signed audit log (premature infrastructure); LLM-assisted single-step recovery (a model on the production path); confidence scoring and `replay --repeat k`.
- **Next, in order:** (1) canaries (dry runs up to the commit) feeding drift events; (2) route scopes per capability, compiled from the requests seen at discovery; (3) a UI Automation surface for one desktop app, to prove the seam; (4) the screencast relay, so a remote person can act in the session; (5) per-tenant overrides with their own approval.
