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

- **Two paths, split at the artifact.** The model runs only in discovery, on a sandbox. Replay never calls it, which keeps production deterministic and cheap and keeps regulated data away from a model. Replay was built first, so discovery had to emit what a proven consumer accepts.
- **Mechanism.** Anthropic's native browser toolset, executed by our own Playwright driver. Grounding comes from the accessibility tree (including frames) first, a screenshot on request, coordinates only as a fallback that is hit-tested back to an element. That works without a clean DOM: ParaBank's unnamed `<select>`s are targeted by id or by the text beside them.
- **Own loop, not the SDK runner:** every step must be policy-gated, recorded, budgeted (40 turns, 10 min, $5), pausable for a person, and testable with a scripted fake model.
- **The `Surface` seam.** Neither path touches Playwright directly. A new surface is a new implementation plus new locator kinds; the artifact doesn't change.
- **Rejected:** a custom tool schema (the trained action space and SDK hooks come free); pixel control for the web (doesn't replay robustly; kept as the desktop design); agent frameworks (they hide the loop, recording and locator decisions); LLM self-healing in production (see Cuts).
- **Simple:** one process, sync Python, Pydantic v2, canonical JSON so approval binds to a hash. 562 offline tests, mypy strict.

## 2. Artifact schema

A calling agent reads `description`, `inputs`, `outputs`, `side_effects` and `outcomes` and never needs `steps` (`understudy describe --as tool` renders them as a tool definition). A reviewer reads the same file with `understudy review`.

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
- **Targets carry a verified fallback chain plus a fingerprint.** Each candidate must resolve to exactly one element, *the same one*, at discovery, with a stability tag and a note saying why. The fingerprint is what a fallback must match, so replay never clicks a look-alike.
- **Every step has its own checkpoint** (`expect`), written in a small signature DSL (`visible`, `absent`, `url`, `selected`, `options_loaded`, `dialog`, `http_status`; `all`/`any`/`not`). The success checkpoint has a **negative control**: it must be false on the start state. ParaBank hides its result panels in the DOM from load, so "text is present" would pass before anything happened.
- **Effect class** (`read` < `input` < `commit`) drives safety. Approval binds to the **content hash**; any edit is a new draft.
- **Lint L001–L011** runs in the compiler, before approval and at load: vacuous success, sensitive literal, a commit that is retried or whose unmatched outcome counts as success, unbound params, unverified targets, policy drift, outputs with no extractor, a base URL or session id baked into a URL or locator, unresolved outcome references.

## 3. Determinism & error handling

- **No sleeps, no model.** Before acting: Playwright auto-wait plus the step's `wait_before`. After acting, an **outcome race** polls every 100 ms in priority order: the app's global states (session expired, app error), the step's `expect` list, then dialogs. Network idle is never a success signal (legacy apps poll). Viewport, locale and timezone are fixed.
- **Resolution.** Locators are tried in order; the first giving one visible element *with a matching fingerprint* wins. A fallback win emits a `drift` event. If all fail, `target_not_found` lists every attempt. Every action is hit-tested first, since Playwright checks pointer interception only for clicks and would otherwise fill a form hidden behind a modal.
- **Three result classes, as the brief asks:**

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
- **Fault injection** (`--inject`, sandbox only) covers each row above; the scenario matrix asserts status, code, commit state and what reached the app. **UI drift is secondary,** surfacing as `drift` events; a real tenant difference already shows ("Account Overview" locally, "Accounts Overview" on the public site).

## 4. Heterogeneity & multi-tenant

*Designed, not built, apart from what is noted.*

- **Surfaces.** The seam is `Surface` plus the `Target` fingerprint. **Legacy web** adds frame paths, table-relative locators (column header plus row key) and label-proximity locators (already used here). **Desktop** adds a UI Automation surface (ControlType→role, Name→name, AutomationId→test id); discovery uses the computer-use toolset, then hit-tests the point to an element for a durable locator. Citrix/RDP is vision-only (OCR anchors plus offsets), the lowest stability tier, so it always gets canaries and review.
- **Reuse.** Three layers: a shared **vendor app profile** (signatures, handlers, policy defaults, redaction); a vendor-scoped **capability** with `compatible_versions`; a **tenant binding** (base URL, secret refs, enabled capabilities, policy *tightening*). Per-tenant differences are overrides keyed by step and locator; an override needs its own approval and can't loosen policy. The effective artifact's hash is recorded on every run. Demonstrated cheaply: the artifact discovered on the public ParaBank replays unchanged on the local one (evidence `02`).
- **Drift.** Passive: drift events aggregated per capability × tenant × app version. Active: scheduled canaries (read-only end to end; commits as a dry run up to the step before). On drift the capability is marked degraded for that tenant and re-discovery on its test environment yields a candidate override diff **for human review**; production is never patched automatically.

## 5. Escalation & handoff

- **Stuck, in discovery:** the model calls `request_human`; the same action on one target three times with no state change; three consecutive errors; no URL or DOM change in 5 actions; 80% of a budget spent without declaring success; a refusal. **In replay:** an unknown blocking state, a commit with an unknown outcome, an approval required.
- **The request** carries the goal or capability with version and hash, the step and intent, the reason, the sanitized URL, a masked live screenshot, recent events and the allowed resolutions.
- **Control transfer: a lease with a fencing token.** States `AGENT` → `AWAITING_HUMAN` (automation paused) → `HUMAN` (claimed) → `AGENT` on resume. Every actuation goes through `Surface.act(…, lease)`; a stale holder raises `ControlLost`, so a paused thread can never act after a handover. Resume and complete need a claim, and only the claimer decides.
- **Same live session.** The person uses the headed browser window the automation was driving, cookies and page state intact. The console is a deliberate mock (a view-only screenshot plus the controls); a remote screencast relay is design only, with its seam (`Control.relay`) in place.
- **What the person did is recorded:** an injected recorder reports trusted click, change and submit events, with descriptors from the same locator synthesizer and values redacted, attributed against the lease log. Input while `HUMAN` is the person's; input at any other time is an `unexpected_input` warning and is never counted. In discovery a person's steps compile into the draft and are flagged for review.
- **Resume** re-checks the paused step (done by the person, ready to run, or escalate again). **Complete** makes replay verify the success checkpoint and read the outputs, refused for a commit not yet sent. **Abort** fails the run. Commits stay blocked at the network while a person holds the session, and earlier inputs are re-verified before the commit if anyone but replay touched the page.
- **Evidence:** `06` (headless: the operator decides, cannot act in the page) and `06b` (headed: the person's click logged as `actor: human`, resume at the paused step, verified success).

## 6. Safety

- **One `PolicyGate`** from a per-app policy a tenant can only tighten: hosts, allowed and denied routes (`/admin.htm`, `/initializeDB.htm` and others denied), browser actions and keys, commit rules. Enforced at five points: the model's tool configuration, the SDK `url_policy` and `confirm` hooks, a browser-level route guard on every request, a per-step check in replay, and lint at load.
- **Risky actions require confirmation, bound to the exact invocation.** Blocking outright would stop the system reaching the confirmation screen; flagging lets money move before anyone looks. It mirrors maker-checker. Risk is judged on the **resolved element**, not the browser action (a purchase is just a click), and the commit route stays blocked at the network except inside a single-use window.
- **Redaction.** Field-aware classification first, regex backstop. Secrets are never written; PII becomes a keyed HMAC tag; financial ids keep at most a third of their digits; URLs lose `jsessionid`; screenshots are masked at capture, DOM snapshots sanitized, page text never logged raw, traces never enabled. Credentials reach only the login step. A scan of `evidence/` for the password, session ids, API key, customer name and every account id found zero hits.
- **Model data.** Discovery runs on sandbox tenants only; production replay sends nothing to a model. Injection defences are layered: page text framed as untrusted, Anthropic's classifiers on tool results, every action through the gate, a human on every commit and artifact.
- **Limits.** Redaction is heuristic and novel PII formats can slip through. Route handlers don't see redirect hops: other hosts are stopped by resolver rules, but a same-host hop to another port or scheme is detected only after it is sent. The console is localhost with a random token, no SSO, and a self-declared operator name. Approval tokens are a CLI flag standing in for a signed grant. A person's recorded actions are reports, not proof (hence the review flag). One compiler gap: the model's free-text success summary, with sample ids, was copied into the draft; the reviewer replaced it, but the compiler is not yet fixed.

## 7. Cuts

- **Left out, deliberately:** multi-tenant overlay resolution and desktop and legacy-web surfaces (designed above, seams in place); a real-time console and the screencast relay (control transfer is real, the surface is a mock); queues, services, a scheduler, egress isolation, a secrets vault and a signed audit log (premature infrastructure); LLM-assisted single-step recovery (a model on the production path, so only as a bounded, policy-checked, recorded fallback); confidence scoring and `replay --repeat k` stability. One public replay was allowed on the shared site and not used.
- **Next, in order:** (1) stop the compiler copying model prose into contract fields, plus a lint that no run-specific value appears in free text; (2) canaries (dry runs up to the commit) feeding drift events; (3) a UI Automation surface for one desktop app, to prove the seam; (4) the screencast relay, so a remote person can act in the session; (5) per-tenant overrides with their own approval.
