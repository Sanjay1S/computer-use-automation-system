"""The hand-written example artifact: ParaBank "Open New Account".

Every locator, option list, heading and sample value here comes from the
recon facts recorded in `apps/parabank/profile.yaml` (gathered by
`scripts/recon.py`). Where recon did not establish something, it is not asserted here:
the two selects carry only their CSS id locator, because the recon snapshot
shows them as unnamed `combobox` nodes (ParaBank labels them with plain <p>
text), and the submit button's fingerprint stops at role+name. The post-commit
error text at s4, unconfirmed by recon, was confirmed later from the page source.

It predates discovery. The shipped capability is the discovered one in
`capabilities/`; this one stays as the test fixture that proves the schema, the
lint rules and the replay engine against a real page's shape. Its JSON export is
`tests/fixtures/example_capability.json`.

It is built in code rather than parsed from JSON so that the lint tests can take
a valid artifact and break exactly one thing about it.
"""

from __future__ import annotations

from understudy.artifact.schema import (
    AppBinding,
    Capability,
    Checkpoint,
    Disposition,
    Expectation,
    Extraction,
    Fingerprint,
    InputSpec,
    Lifecycle,
    Locator,
    LocatorStats,
    OutcomeRef,
    OutcomeSpec,
    OutputSpec,
    ParamValue,
    Provenance,
    RecoveryPolicy,
    Requirements,
    RetryPolicy,
    SideEffects,
    Step,
    Target,
    Verification,
)
from understudy.artifact.signatures import (
    AllCondition,
    NotCondition,
    OptionsLoadedCondition,
    SelectedCondition,
    UrlCondition,
    VisibleCondition,
)

RUN_ID = "handwritten-m2"

VERIFIED = Verification(unique=True, same_element=True, run_id=RUN_ID)


def example_capability() -> Capability:
    return Capability(
        id="parabank.open_account",
        version="1.0.0",
        title="Open a new ParaBank account",
        description=(
            "Opens a new CHECKING or SAVINGS account for the signed-in ParaBank "
            "customer, funded by transferring the minimum opening deposit from one "
            "of their existing accounts, and returns the new account's id. The "
            "funding account must belong to the signed-in customer."
        ),
        lifecycle=Lifecycle(status="draft"),
        app=AppBinding(
            vendor_app="parabank",
            surface="web",
            entry_route="/openaccount.htm",
            compatible_versions=["3.*"],
        ),
        requires=Requirements(session="parabank.customer"),
        inputs=[
            InputSpec(
                name="account_type",
                type="enum",
                enum=["CHECKING", "SAVINGS"],
                description="The kind of account to open.",
                example="SAVINGS",
            ),
            InputSpec(
                name="funding_account_id",
                type="string",
                pattern=r"^\d+$",
                sensitivity="financial_id",
                description=(
                    "The existing account the opening deposit comes from. Must be one "
                    "of the signed-in customer's own accounts."
                ),
                example="12345",
            ),
        ],
        outputs=[
            OutputSpec(
                name="new_account_id",
                type="string",
                pattern=r"^\d+$",
                sensitivity="financial_id",
                source_step="s5_read_new_id",
                description="The id of the account that was opened.",
            )
        ],
        side_effects=SideEffects(
            cls="commit",
            idempotent=False,
            approval="per_invocation",
            description=(
                "Creates a real account and moves the minimum opening deposit out of "
                "the funding account. Running it twice opens two accounts."
            ),
        ),
        steps=[_s1_open_form(), _s2_type(), _s3_funding(), _s4_submit(), _s5_extract()],
        success=Checkpoint(
            description=("The confirmation panel is showing and it carries a numeric account id."),
            signature=AllCondition(
                all=[
                    VisibleCondition(text="Account Opened!"),
                    VisibleCondition(css="#newAccountId"),
                ]
            ),
            negative_control=AllCondition(
                all=[
                    VisibleCondition(role="heading", name="Open New Account"),
                    NotCondition(not_=VisibleCondition(text="Account Opened!")),
                ]
            ),
        ),
        outcomes=[
            OutcomeSpec(
                code="funding_account_not_found",
                detected_by="s3_funding_account",
                description=(
                    "The funding account list loaded, but it does not contain the "
                    "requested account: it is not one of this customer's accounts. "
                    "Nothing was committed."
                ),
            )
        ],
        recovery=RecoveryPolicy(
            allowed=["retry_transient", "reauthenticate", "dismiss_interstitial"],
            budget=3,
        ),
        provenance=Provenance(
            discovery_run_id=RUN_ID,
            recorded_against="parabank-local",
            authored="hand_written",
            goal_template="Open a new {account_type} account funded from {funding_account_id}",
            locator_stats=LocatorStats(targets=4, locators=5, verified_unique=5),
        ),
    )


def _s1_open_form() -> Step:
    return Step(
        id="s1_open_form",
        intent="Go to the account opening form",
        action="navigate",
        effect="read",
        url="{base_url}/openaccount.htm",
        reentry_point=True,
        timeout_ms=15_000,
        retry=RetryPolicy(max_attempts=2, backoff_ms=[500, 1000]),
        expect=[
            Expectation(
                description="The form is showing",
                when=AllCondition(
                    all=[
                        UrlCondition(path_regex=r"/openaccount\.htm$"),
                        VisibleCondition(role="heading", name="Open New Account"),
                    ]
                ),
                then=Disposition(do="continue"),
            )
        ],
    )


def _s2_type() -> Step:
    return Step(
        id="s2_account_type",
        intent="Choose the account type",
        action="select",
        effect="input",
        value=ParamValue(param="account_type"),
        target=Target(
            description="The account-type dropdown on the open-account form",
            # The recon snapshot shows a bare `combobox` with no accessible name:
            # ParaBank labels these selects with plain <p> text, not <label for>.
            # So there is no role+name or label locator to offer here, and the
            # hand-written JSP id is the only verified handle.
            fingerprint=Fingerprint(role="combobox", tag="select"),
            locators=[
                Locator(
                    by="css",
                    value="select#type",
                    stability="medium",
                    verified=VERIFIED,
                    note=(
                        "The id is hand-written in the JSP, not generated, so it survives "
                        "redeploys; it would not survive a rewrite of the page. Medium, "
                        "not high: it is the only candidate, with no fallback behind it."
                    ),
                ),
            ],
        ),
        expect=[
            Expectation(
                description="The dropdown holds the requested type",
                # Matches the option's label or value: the enum speaks labels
                # (SAVINGS), the page stores values (1).
                when=SelectedCondition(target="select#type", param="account_type"),
                then=Disposition(do="continue"),
            )
        ],
    )


def _s3_funding() -> Step:
    return Step(
        id="s3_funding_account",
        intent="Choose the account the opening deposit comes from",
        action="select",
        effect="input",
        value=ParamValue(param="funding_account_id"),
        # The list arrives by AJAX after the page is interactive, so acting before it
        # has loaded would select nothing and mis-report it as 'account not found'.
        wait_before=OptionsLoadedCondition(target="select#fromAccountId", min=1),
        on_target_absent=OutcomeRef(outcome="funding_account_not_found"),
        target=Target(
            description="The funding account dropdown, AJAX-loaded after page load",
            # Same as s2: recon shows an unnamed `combobox`, so no label locator exists.
            fingerprint=Fingerprint(role="combobox", tag="select"),
            locators=[
                Locator(
                    by="css",
                    value="select#fromAccountId",
                    stability="medium",
                    verified=VERIFIED,
                    note=(
                        "Hand-written JSP id; the AJAX handler targets it by this id too, "
                        "so the id cannot drift without the page's own script breaking."
                    ),
                ),
            ],
        ),
        expect=[
            Expectation(
                description="The requested funding account is selected",
                when=SelectedCondition(target="select#fromAccountId", param="funding_account_id"),
                then=Disposition(do="continue"),
            )
        ],
    )


def _s4_submit() -> Step:
    return Step(
        id="s4_submit",
        intent="Submit the form. This opens the account and moves the deposit.",
        action="click",
        effect="commit",
        # No retry and no re-entry: a second click opens a second account.
        retry=RetryPolicy(max_attempts=0),
        timeout_ms=30_000,
        target=Target(
            description=(
                "The 'Open New Account' submit button. The left-hand nav has a link of "
                "exactly the same name, so role is what separates them."
            ),
            # Recon recorded `button "Open New Account"` and nothing more, so the
            # fingerprint stops there. A tag or input type guessed here would make
            # correct resolutions fail identity verification in replay.
            fingerprint=Fingerprint(role="button", name="Open New Account"),
            locators=[
                Locator(
                    by="role",
                    value={"role": "button", "name": "Open New Account", "exact": True},
                    stability="high",
                    verified=VERIFIED,
                    note=(
                        "Role+name is unique: recon confirmed the heading, the nav link and "
                        "this button all read 'Open New Account', and only role tells "
                        "them apart. A name-only or text locator would match three "
                        "elements and is therefore not offered as a fallback."
                    ),
                ),
                Locator(
                    by="css",
                    # An <input type="button">, not a submit (page source): the form has no
                    # action and the page script POSTs createAccount on click.
                    value="input[type='button'][value='Open New Account']",
                    stability="medium",
                    verified=VERIFIED,
                    note="The value attribute is the button's label; it moves with it.",
                ),
            ],
        ),
        expect=[
            Expectation(
                description="The account was opened",
                when=VisibleCondition(text="Account Opened!"),
                then=Disposition(do="continue"),
            ),
            # Confirmed from the page source: this text sits in the hidden
            # #openAccountError panel, which the page script shows when the
            # createAccount request fails. `visible` matters: the panel is in the DOM,
            # hidden, on every load.
            Expectation(
                description=(
                    "The app errored after the click. The deposit may or may not have "
                    "moved, so this goes to a human rather than to a retry."
                ),
                when=VisibleCondition(text="An internal error has occurred and has been logged."),
                then=Disposition(do="escalate", code="app_error_after_commit"),
            ),
        ],
        # Neither signature appearing within the timeout is the dangerous case: it
        # inherits escalate(commit_outcome_unknown) from the commit effect.
    )


def _s5_extract() -> Step:
    return Step(
        id="s5_read_new_id",
        intent="Read the new account id off the confirmation panel",
        action="extract",
        effect="read",
        retry=RetryPolicy(max_attempts=2, backoff_ms=[250, 500]),
        target=Target(
            description="The new account id on the confirmation panel",
            fingerprint=Fingerprint(tag="a"),
            locators=[
                Locator(
                    by="css",
                    value="#newAccountId",
                    stability="high",
                    verified=VERIFIED,
                    note="The only element on the page carrying the id; recon read one from it.",
                )
            ],
        ),
        extract=Extraction(
            output="new_account_id", read="text", parse="regex", regex=r"^\s*(\d+)\s*$"
        ),
        expect=[
            Expectation(
                description="A numeric account id is present",
                when=VisibleCondition(css="#newAccountId"),
                then=Disposition(do="succeed"),
            )
        ],
    )
