"""The signature DSL: small, deterministic page-state predicates.

A signature is what a step waits for, what an outcome is detected by, and what
the success checkpoint asserts. Evaluation (M4) uses non-waiting Playwright
checks inside the outcome race, so every condition here must be answerable from
a single synchronous look at the page.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UrlCondition(_Node):
    """Matches the current URL's path. Never the host: that is a tenant binding."""

    kind: Literal["url"] = "url"
    path_regex: str


class _ElementMatcher(_Node):
    """The matcher fields shared by `visible` and `absent`. Subclasses set `kind`."""

    kind: str

    MATCHERS: ClassVar[tuple[str, ...]] = (
        "role",
        "name",
        "name_regex",
        "text",
        "text_regex",
        "css",
    )

    role: str | None = None
    name: str | None = None
    name_regex: str | None = None
    text: str | None = None
    text_regex: str | None = None
    css: str | None = None
    within: str | None = None
    exact: bool = True

    @model_validator(mode="after")
    def _at_least_one_matcher(self) -> Self:
        if not any(getattr(self, field) is not None for field in self.MATCHERS):
            raise ValueError(f"{self.kind} needs at least one of: " + ", ".join(self.MATCHERS))
        return self


class VisibleCondition(_ElementMatcher):
    """An element is present and visible. At least one of the matchers is required."""

    kind: Literal["visible"] = "visible"


class AbsentCondition(_ElementMatcher):
    """No element matches. The inverse of `visible`, and not the same as `not visible`:
    this one is true when nothing matches, where `not` negates whatever the inner
    condition decided."""

    kind: Literal["absent"] = "absent"


class SelectedCondition(_Node):
    """A select or radio group holds a value: either a literal or an input param."""

    kind: Literal["selected"] = "selected"
    target: str
    value: str | None = None
    param: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> SelectedCondition:
        if (self.value is None) == (self.param is None):
            raise ValueError("selected needs exactly one of: value, param")
        return self


class OptionsLoadedCondition(_Node):
    """A select has at least `min` real options. The readiness check for AJAX lists."""

    kind: Literal["options_loaded"] = "options_loaded"
    target: str
    min: int = Field(default=1, ge=1)


class DialogCondition(_Node):
    """A native dialog (alert, confirm, prompt) is pending."""

    kind: Literal["dialog"] = "dialog"
    message_regex: str | None = None


class HttpStatusCondition(_Node):
    """The main frame's last response status falls in a range. 5xx drives transient
    recovery; it is deliberately not inferred from page text."""

    kind: Literal["http_status"] = "http_status"
    min: int = Field(ge=100, le=599)
    max: int = Field(ge=100, le=599)

    @model_validator(mode="after")
    def _ordered(self) -> HttpStatusCondition:
        if self.min > self.max:
            raise ValueError("http_status min must be <= max")
        return self


class AllCondition(_Node):
    kind: Literal["all"] = "all"
    all: list[Signature] = Field(min_length=1)


class AnyCondition(_Node):
    kind: Literal["any"] = "any"
    any: list[Signature] = Field(min_length=1)


class NotCondition(_Node):
    kind: Literal["not"] = "not"
    not_: Signature = Field(alias="not")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Signature = Annotated[
    UrlCondition
    | VisibleCondition
    | AbsentCondition
    | SelectedCondition
    | OptionsLoadedCondition
    | DialogCondition
    | HttpStatusCondition
    | AllCondition
    | AnyCondition
    | NotCondition,
    Field(discriminator="kind"),
]

ELEMENT_KINDS = frozenset({"visible", "absent", "selected", "options_loaded", "dialog"})


def walk(signature: Signature) -> list[Signature]:
    """Every node in the tree, parents before children."""
    nodes: list[Signature] = [signature]
    if isinstance(signature, AllCondition):
        for child in signature.all:
            nodes.extend(walk(child))
    elif isinstance(signature, AnyCondition):
        for child in signature.any:
            nodes.extend(walk(child))
    elif isinstance(signature, NotCondition):
        nodes.extend(walk(signature.not_))
    return nodes


def has_element_condition(signature: Signature) -> bool:
    """True when the signature asserts something about the page's elements.

    A checkpoint built only from `url` conditions passes on any page the route
    happens to render, including an error page, which is why lint L001 requires
    one of these.
    """
    return any(node.kind in ELEMENT_KINDS for node in walk(signature))


def params_used(signature: Signature) -> set[str]:
    """Input params referenced anywhere in the tree."""
    return {
        node.param
        for node in walk(signature)
        if isinstance(node, SelectedCondition) and node.param is not None
    }


AllCondition.model_rebuild()
AnyCondition.model_rebuild()
NotCondition.model_rebuild()
