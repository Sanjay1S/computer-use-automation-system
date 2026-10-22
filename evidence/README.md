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
| `07_replay_commit_unknown/` | the commit response is dropped | `--inject drop_commit_response` | `needs_human: commit_outcome_unknown` | `needs_human: commit_outcome_unknown`, `commit_state=unknown`, nothing retried, exit 5 |

## Known gaps

- **`06` is not the full demo.** This machine has no display (plan F22), so the browser was headless. The operator decided in the console but could not act in the page, and `human_actions` is 0. What it shows: the lease passes to the operator and back, the run resumes from a re-entry point, and the result is a verified success. What it does not show: a person's clicks recorded as `actor: human`. The headed version still needs a machine with a display (README, "Operator handoff"). The offline suite covers the human-action path (`test_replay_handoff.py`, `test_human_recorder.py`).
- **No public replay.** Plan §2 D1 allows one replay on the public instance; it wasn't used. `02` already shows the public-discovered artifact running on another instance.
- **`01`'s `draft.json` is left out.** The compiler copied the model's free-text success summary into `success.description`, which put the sample funding and new account ids in it. The reviewer replaced it with a generic sentence before approval, so `artifact/` is the reviewed version. `artifact/review_as_compiled.txt` is the review summary as the compiler produced it.

## Leak scan

`grep -rI` over this folder for the ParaBank password, `jsessionid`, `sk-ant`, the customer's name, and every sample or newly opened account id: 0 files each. Screenshots were checked by eye on two samples (the new account number, the funding select and the account holder are masked).
