import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def use_test_database():
    """Redirect the application to the test database for the whole run.

    autouse=True so that no test can reach the real one by forgetting to ask.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if url is not None:
        os.environ["DATABASE_URL"] = url


def pytest_collection_modifyitems(items):
    """Skip integration tests when there is no test database to point them at.

    This is where the protection actually belongs: without TEST_DATABASE_URL,
    integration tests are skipped outright, so none of them can fall back to
    DATABASE_URL from .env and hit the real database.
    """
    if os.environ.get("TEST_DATABASE_URL") is not None:
        return

    skip_integration = pytest.mark.skip(reason="TEST_DATABASE_URL is not set")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)