from collections.abc import Iterator

import pytest
from playwright.sync_api import Browser, Page, sync_playwright


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        timezone_id="UTC",
        reduced_motion="reduce",
        service_workers="block",
    )
    page = context.new_page()
    yield page
    context.close()
