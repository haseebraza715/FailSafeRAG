"""Shared terminal no-recovery baseline checks."""

from typing import Any


def terminal_baseline(config: dict[str, Any], summary: dict[str, Any], pilot_id: str) -> bool:
    """Apply the comparison's run and blocker criteria for a terminal baseline."""
    identity = config.get("identity", config)
    blockers = summary.get("valid_baseline_blockers", [])
    return (summary.get("run_state") == "complete" and summary.get("pilot_id") == pilot_id
            and config.get("mode") == "live" and identity.get("run_kind") == "development_pilot"
            and identity.get("pilot_manifest", {}).get("canonical") is True
            and all(isinstance(b, str) and b.endswith("question(s) are execution_failed") for b in blockers))


def terminal_prediction(row: dict[str, Any]) -> bool:
    """Apply the comparison's terminal prediction rule."""
    return (not row.get("unserved") and not row.get("awaiting_reconciliation")
            and row.get("status") in ("answered", "no_evidence", "execution_failed"))
