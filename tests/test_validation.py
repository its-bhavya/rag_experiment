import json
import pathlib

from src.experiments.validate_experiments import run_all_validation_checks

def test_validation_passes():
    report = run_all_validation_checks()
    # Ensure overall validation passed flag exists
    assert isinstance(report, dict)
    assert report.get("validation_passed") is True, f"Validation failed: {json.dumps(report, indent=2)}"
