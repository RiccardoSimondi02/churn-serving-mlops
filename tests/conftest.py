import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def use_test_database():
    """Redirect the application to the test database for the whole run.

    autouse=True so that no test can reach the real one by forgetting to ask.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if url is None:
        pytest.skip("TEST_DATABASE_URL is not set", allow_module_level=True)
    os.environ["DATABASE_URL"] = url