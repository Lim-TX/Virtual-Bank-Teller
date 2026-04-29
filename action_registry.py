"""
Action Registry — single source of truth for all clickable action surfaces.

Supersedes the legacy _ql_queries dict and _ACTION_KEYS set in app.py.

Each entry defines:
  surface  : "workflow" (full main-panel task page) or "modal" (st.dialog)
  title    : human-facing label used in buttons and headings
  mock_key : key into MOCK_ACCOUNT for the data this surface needs (or None)
"""

from typing import TypedDict


class ActionEntry(TypedDict):
    surface: str      # "workflow" | "modal"
    title: str
    mock_key: str | None


ACTION_REGISTRY: dict[str, ActionEntry] = {
    # ── Transactional workflow surfaces ─────────────────────────────────────
    "send_money": {
        "surface": "workflow",
        "title": "Send Money",
        "mock_key": "send_money",
    },
    "pay_card": {
        "surface": "workflow",
        "title": "Pay Card",
        "mock_key": "pay_card",
    },
    "pay_loan": {
        "surface": "workflow",
        "title": "Pay Loan / Financing",
        "mock_key": "pay_loan",
    },
    "pay_bill": {
        "surface": "workflow",
        "title": "Pay Bill",
        "mock_key": "pay_bill",
    },
    # ── Informational modal surfaces ─────────────────────────────────────────
    "promotions": {
        "surface": "modal",
        "title": "Promotions",
        "mock_key": "promotions",
    },
    "mailbox": {
        "surface": "modal",
        "title": "Mailbox",
        "mock_key": "recent_transactions",
    },
    "account_summary": {
        "surface": "modal",
        "title": "Account Summary",
        "mock_key": None,
    },
    "maintenance": {
        "surface": "modal",
        "title": "System Maintenance",
        "mock_key": "maintenance_notice",
    },
}


def get_action(key: str) -> ActionEntry:
    """Return the registry entry for *key*, raising KeyError if not found."""
    return ACTION_REGISTRY[key]


def get_all_action_keys() -> set[str]:
    """Return all registered action keys."""
    return set(ACTION_REGISTRY.keys())


def is_workflow(key: str) -> bool:
    """Return True when *key* maps to a full workflow surface."""
    entry = ACTION_REGISTRY.get(key)
    return entry is not None and entry["surface"] == "workflow"


def is_modal(key: str) -> bool:
    """Return True when *key* maps to a modal surface."""
    entry = ACTION_REGISTRY.get(key)
    return entry is not None and entry["surface"] == "modal"
