from __future__ import annotations

import pytest


pytestmark = pytest.mark.legacy


def test_legacy_api_operation_cycle_contract_is_retired() -> None:
    pytest.skip(
        "legacy pipeline/CriticAgent/policy-blocking API test retired; current API "
        "coverage lives in operation run summary and stage-agent main-path tests"
    )
