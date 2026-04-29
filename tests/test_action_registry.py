"""Tests for action_registry.py — written BEFORE implementation (TDD Gate)."""
import pytest
from action_registry import (
    get_action,
    get_all_action_keys,
    is_modal,
    is_workflow,
    ACTION_REGISTRY,
)


class TestActionRegistry:
    # ── Registry completeness ──────────────────────────────────────────────
    def test_contains_all_four_workflow_keys(self):
        keys = get_all_action_keys()
        assert "send_money" in keys
        assert "pay_card" in keys
        assert "pay_loan" in keys
        assert "pay_bill" in keys

    def test_contains_informational_keys(self):
        keys = get_all_action_keys()
        assert "promotions" in keys
        assert "mailbox" in keys
        assert "account_summary" in keys
        assert "maintenance" in keys

    def test_each_entry_has_required_fields(self):
        for key, entry in ACTION_REGISTRY.items():
            assert "surface" in entry, f"{key} missing 'surface'"
            assert "title" in entry, f"{key} missing 'title'"
            assert entry["surface"] in ("workflow", "modal"), (
                f"{key} has invalid surface type: {entry['surface']}"
            )

    # ── get_action ─────────────────────────────────────────────────────────
    def test_get_action_returns_entry_for_known_key(self):
        entry = get_action("send_money")
        assert entry is not None
        assert entry["title"] == "Send Money"

    def test_get_action_raises_for_unknown_key(self):
        with pytest.raises(KeyError):
            get_action("nonexistent_key")

    def test_get_action_pay_card_has_workflow_surface(self):
        assert get_action("pay_card")["surface"] == "workflow"

    def test_get_action_promotions_has_modal_surface(self):
        assert get_action("promotions")["surface"] == "modal"

    # ── is_modal / is_workflow ─────────────────────────────────────────────
    def test_is_workflow_true_for_transactional_keys(self):
        for key in ("send_money", "pay_card", "pay_loan", "pay_bill"):
            assert is_workflow(key) is True, f"{key} should be workflow"

    def test_is_modal_true_for_informational_keys(self):
        for key in ("promotions", "mailbox", "account_summary", "maintenance"):
            assert is_modal(key) is True, f"{key} should be modal"

    def test_is_workflow_false_for_modal_key(self):
        assert is_workflow("promotions") is False

    def test_is_modal_false_for_workflow_key(self):
        assert is_modal("send_money") is False

    def test_is_workflow_false_for_unknown_key(self):
        assert is_workflow("unknown") is False

    def test_is_modal_false_for_unknown_key(self):
        assert is_modal("unknown") is False
