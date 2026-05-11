import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("thorough", max_examples=500, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("default", max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.too_slow])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Append the assertion message to the xfail reason so `-rx` shows actual numbers."""
    outcome = yield
    report = outcome.get_result()
    wasxfail = getattr(report, "wasxfail", None)
    if report.when == "call" and wasxfail is not None and call.excinfo is not None:
        msg = str(call.excinfo.value).splitlines()[0] if call.excinfo.value else ""
        if msg and msg not in wasxfail:
            report.wasxfail = f"{wasxfail} — {msg}"
