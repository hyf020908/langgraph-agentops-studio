from __future__ import annotations

import os

from services.config import load_settings


def test_reporting_detail_configuration_from_environment() -> None:
    previous = dict(os.environ)
    try:
        os.environ["REPORT_DETAIL_LEVEL"] = "compact"
        os.environ["REPORT_MAX_FINDINGS_IN_CONCLUSION"] = "2"
        os.environ["REPORT_MAX_EVIDENCE_IN_CONCLUSION"] = "3"
        os.environ["REPORT_MAX_RISKS_IN_CONCLUSION"] = "1"
        os.environ["REPORT_INCLUDE_NEXT_ACTIONS"] = "false"

        load_settings.cache_clear()
        settings = load_settings()

        assert settings.reporting.detail_level == "compact"
        assert settings.reporting.max_findings_in_conclusion == 2
        assert settings.reporting.max_evidence_in_conclusion == 3
        assert settings.reporting.max_risks_in_conclusion == 1
        assert settings.reporting.include_next_actions is False
    finally:
        os.environ.clear()
        os.environ.update(previous)
        load_settings.cache_clear()
