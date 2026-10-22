# Evidence

Runs from 2026-10-22. Each folder is a curated copy of a run folder (`runs/<run_id>/`, already redacted and masked). Nothing was edited except as noted for `01`.

The artifact was discovered once on the public ParaBank instance and approved by content hash. Every replay below uses that approved artifact on the **local** tenant, which is the multi-tenant claim: one capability, two instances, no change to the artifact.

`artifact/parabank.open_account@1.0.0.json` is the approved artifact (`sha256:c7114ccc5cfabf48f8373f26880980fe6fcb94abd30cf169c12dd90b50994771`, reviewer Sanjay Srinivasa). Lint is clean.

| Folder | Run | Command (after `set -a; . ./.env; set +a`) | Expected | Actual |
|---|---|---|---|---|
| `01_discovery_open_account/` | real model run, `claude-opus-5-5`, public ParaBank, the commit approved in the console | `understudy discover goals/open_account.yaml --tenant parabank-public` | draft compiled, lint clean | `compiled`, `declared_success`, 8 turns, 21.8 s, $0.098, lint clean, one approval (`commit_approval`, approved) |
| `02_replay_success/` | different inputs, other tenant | `understudy replay capabilities/parabank.open_account@1.0.0.json --tenant parabank-local --approve --input account_type=CHECKING --input funding_account_id=13344` | `succeeded` + `new_account_id` | `succeeded`, `new_account_id` returned, `commit_state=committed`, exit 0 |
| `03_replay_business_outcome/` | funding account not owned | same, with `--input account_type=SAVINGS --input funding_account_id=99999` | `business_outcome` | `business_outcome: funding_account_id_not_found`, `commit_state=not_attempted`, exit 4 |
| `04_replay_recovered/` | session expires before the commit | same inputs as 02 plus `--inject session_expired_at=s3` | `succeeded`, `reauthenticate` | `succeeded`, `recoveries=[reauthenticate]`, committed, exit 0 |
| `05_replay_hard_failure/` | the commit button is renamed | `--inject rename_target:s4` | `failed: target_not_found` | `failed: automation/target_not_found`, every locator attempt listed, failure bundle, `commit_state=not_attempted`, exit 1 |
| `06_replay_handoff/` | an unknown modal blocks `s3`; an operator takes over in the console | `--on-stuck escalate --inject unknown_modal_at:s3` | `succeeded`, handoff logged | `succeeded`, one `takeover` intervention (`unknown_blocking_state`) claimed by the operator and resumed from `s1_go_openaccount`, committed |
| `06b_replay_handoff_headed/` | the same fault, **headed**: the operator claims the session, closes the modal in the real browser window, then resumes at the paused step | same command with `--headed`, resumed at `s3` | `succeeded`, the person's click logged as `actor: human` | `succeeded`, `completed_by=automation`, one `takeover` (`resolution=resumed`, `human_actions=1`, `unexpected_inputs=0`, note "I closed the pop up"). Events 8–12 show `AWAITING_HUMAN` → claim (`HUMAN`) → `human` click on `button 'OK'` → decision → `AGENT`; automation then ran `s3`–`s5`. Committed, exit 0 |
| `07_replay_commit_unknown/` | the commit response is dropped | `--inject drop_commit_response` | `needs_human: commit_outcome_unknown` | `needs_human: commit_outcome_unknown`, `commit_state=unknown`, nothing retried, exit 5 |

## Known gaps

- **`06` is the headless handoff; `06b` is the real one.** The server has no display (plan F22), so `06` ran headless: the operator decided in the console but could not act in the page (`human_actions` is 0, and the run resumed from `s1`). `06b` was run headed on a laptop against the same ParaBank, over an SSH tunnel, and shows a person's click in the live window recorded as `actor: human`. A first headed attempt was discarded, because the operator clicked before claiming (that input is correctly logged as `unexpected_input`, not as the person's action) and then navigated the automation window to the console, which the policy guard refused.
- **The operator surface is a mock at a clean seam.** The console is a small localhost page with a masked screenshot that refreshes every 2 s, view-only. The person acts in the browser window itself; a remote screencast relay is design only (`Control.relay` is the seam).
- **No public replay.** Plan §2 D1 allows one replay on the public instance; it wasn't used. `02` already shows the public-discovered artifact running on another instance.
- **`01`'s `draft.json` is left out.** The compiler copied the model's free-text success summary into `success.description`, which put the sample funding and new account ids in it. The reviewer replaced it with a generic sentence before approval, so `artifact/` is the reviewed version. `artifact/review_as_compiled.txt` is the review summary as the compiler produced it.

## Leak scan

`grep -rI` over this folder (including `06b`) for the ParaBank password, `jsessionid`, `sk-ant`, the customer's name, and every sample or newly opened account id: 0 files each. Screenshots were checked by eye on two samples (the new account number, the funding select and the account holder are masked).
