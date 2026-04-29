"""
Tests for workflow helper functions in workflows/*.py — written BEFORE implementation (TDD Gate).

These functions are pure-data helpers (they do NOT call st.*).
Streamlit rendering is integration-tested by smoke-running the app.
"""
import pytest
import sys
import os

# Ensure repo root is on sys.path so imports work from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.send_money import build_send_money_context
from workflows.pay_card import build_pay_card_context
from workflows.pay_loan import build_pay_loan_context
from workflows.pay_bill import build_pay_bill_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as _app
_parse_send_money_prefill = _app._parse_send_money_prefill
_parse_pay_card_prefill   = _app._parse_pay_card_prefill
_parse_pay_loan_prefill   = _app._parse_pay_loan_prefill
_parse_pay_bill_prefill   = _app._parse_pay_bill_prefill
_detect_emotion           = _app._detect_emotion
_EMOTION_PROSODY          = _app._EMOTION_PROSODY


# ── Shared minimal MOCK_ACCOUNT fixture ───────────────────────────────────────
@pytest.fixture()
def mock_account():
    return {
        "name": "Test User",
        "account_number": "0000-0000-0001",
        "balance": "RM 10,000.00",
        "savings_account": {"account_number": "0000-0000-0002", "balance": "RM 5,000.00"},
        "credit_card": {
            "number": "**** 9999",
            "outstanding_balance": "RM 500.00",
            "minimum_payment": "RM 25.00",
            "due_date": "30 Apr 2026",
        },
        "loan": {
            "type": "Personal Financing",
            "outstanding_balance": "RM 20,000.00",
            "monthly_instalment": "RM 600.00",
            "next_due_date": "1 May 2026",
            "remaining_tenure": "3 years",
            "payment_options": ["Full instalment", "Partial payment"],
        },
        "send_money": {
            "daily_limit": "RM 50,000.00",
            "per_transaction_limit": "RM 10,000.00",
            "remaining_today": "RM 50,000.00",
            "fee": "Free",
            "transfer_types": ["DuitNow", "IBG"],
            "supported_banks": ["Maybank", "CIMB"],
        },
        "pay_card": {
            "payment_options": [
                {"option": "Minimum payment", "amount": "RM 25.00"},
                {"option": "Full balance", "amount": "RM 500.00"},
            ],
            "payment_source": "Current Account 0000-0000-0001",
        },
        "pay_loan": {
            "payment_options": [
                {"option": "Monthly instalment", "amount": "RM 600.00"},
            ],
            "payment_source": "Current Account 0000-0000-0001",
            "early_settlement_fee": "1%",
        },
        "pay_bill": {
            "registered_billers": [
                {"biller": "TNB", "account_ref": "111", "amount_due": "RM 50.00", "due": "20 Apr 2026"},
            ],
            "payment_source": "Current Account 0000-0000-0001",
        },
    }


# ── send_money ─────────────────────────────────────────────────────────────────
class TestBuildSendMoneyContext:
    def test_returns_dict(self, mock_account):
        ctx = build_send_money_context(mock_account)
        assert isinstance(ctx, dict)

    def test_contains_limits(self, mock_account):
        ctx = build_send_money_context(mock_account)
        assert "daily_limit" in ctx
        assert "per_transaction_limit" in ctx
        assert "remaining_today" in ctx

    def test_contains_fee(self, mock_account):
        ctx = build_send_money_context(mock_account)
        assert "fee" in ctx

    def test_contains_source_accounts(self, mock_account):
        ctx = build_send_money_context(mock_account)
        assert "source_accounts" in ctx
        assert len(ctx["source_accounts"]) >= 1

    def test_contains_transfer_types(self, mock_account):
        ctx = build_send_money_context(mock_account)
        assert "transfer_types" in ctx
        assert len(ctx["transfer_types"]) > 0


# ── pay_card ───────────────────────────────────────────────────────────────────
class TestBuildPayCardContext:
    def test_returns_dict(self, mock_account):
        ctx = build_pay_card_context(mock_account)
        assert isinstance(ctx, dict)

    def test_contains_outstanding_balance(self, mock_account):
        ctx = build_pay_card_context(mock_account)
        assert "outstanding_balance" in ctx

    def test_contains_payment_options(self, mock_account):
        ctx = build_pay_card_context(mock_account)
        assert "payment_options" in ctx
        assert len(ctx["payment_options"]) > 0

    def test_contains_payment_source(self, mock_account):
        ctx = build_pay_card_context(mock_account)
        assert "payment_source" in ctx


# ── pay_loan ───────────────────────────────────────────────────────────────────
class TestBuildPayLoanContext:
    def test_returns_dict(self, mock_account):
        ctx = build_pay_loan_context(mock_account)
        assert isinstance(ctx, dict)

    def test_contains_outstanding_balance(self, mock_account):
        ctx = build_pay_loan_context(mock_account)
        assert "outstanding_balance" in ctx

    def test_contains_monthly_instalment(self, mock_account):
        ctx = build_pay_loan_context(mock_account)
        assert "monthly_instalment" in ctx

    def test_contains_payment_options(self, mock_account):
        ctx = build_pay_loan_context(mock_account)
        assert "payment_options" in ctx

    def test_contains_next_due_date(self, mock_account):
        ctx = build_pay_loan_context(mock_account)
        assert "next_due_date" in ctx


# ── pay_bill ───────────────────────────────────────────────────────────────────
class TestBuildPayBillContext:
    def test_returns_dict(self, mock_account):
        ctx = build_pay_bill_context(mock_account)
        assert isinstance(ctx, dict)

    def test_contains_registered_billers(self, mock_account):
        ctx = build_pay_bill_context(mock_account)
        assert "registered_billers" in ctx
        assert len(ctx["registered_billers"]) > 0

    def test_each_biller_has_required_fields(self, mock_account):
        ctx = build_pay_bill_context(mock_account)
        for biller in ctx["registered_billers"]:
            assert "biller" in biller
            assert "amount_due" in biller
            assert "due" in biller

    def test_contains_payment_source(self, mock_account):
        ctx = build_pay_bill_context(mock_account)
        assert "payment_source" in ctx


# ── _parse_pay_card_prefill ────────────────────────────────────────────────────
class TestParsePayCardPrefill:
    def test_minimum_payment_keyword(self):
        r = _parse_pay_card_prefill("I want to pay the minimum")
        assert r is not None
        assert "Minimum payment" in r["option"]

    def test_full_balance_keywords(self):
        r = _parse_pay_card_prefill("pay full outstanding balance")
        assert r is not None
        assert "Full outstanding balance" in r["option"]

    def test_custom_amount(self):
        r = _parse_pay_card_prefill("I want to pay RM 200 on my card")
        assert r is not None
        assert "Custom amount" in r["option"]
        assert r["custom"] == 200.0

    def test_returns_none_if_no_match(self):
        assert _parse_pay_card_prefill("I want to pay my card") is None

    def test_full_balance_pay_full(self):
        r = _parse_pay_card_prefill("pay full balance on my card")
        assert r is not None
        assert "Full outstanding balance" in r["option"]


# ── _parse_pay_loan_prefill ───────────────────────────────────────────────────
class TestParsePayLoanPrefill:
    def test_monthly_instalment(self):
        r = _parse_pay_loan_prefill("pay my monthly instalment")
        assert r is not None
        assert "Monthly instalment" in r["option"]

    def test_full_settlement(self):
        r = _parse_pay_loan_prefill("I want full settlement of the loan")
        assert r is not None
        assert "Full settlement" in r["option"]

    def test_partial_payment_with_amount(self):
        r = _parse_pay_loan_prefill("partial payment of RM 1000")
        assert r is not None
        assert "Partial payment" in r["option"]
        assert r["partial"] == 1000.0

    def test_partial_payment_without_amount(self):
        r = _parse_pay_loan_prefill("I want to make a partial payment")
        assert r is not None
        assert "Partial payment" in r["option"]
        assert "partial" not in r

    def test_returns_none_if_no_match(self):
        assert _parse_pay_loan_prefill("pay my loan") is None


# ── _parse_pay_bill_prefill ───────────────────────────────────────────────────
class TestParsePayBillPrefill:
    def test_tnb_keyword(self):
        r = _parse_pay_bill_prefill("pay my TNB electricity bill")
        assert r is not None
        assert r["biller_name"] == "Tenaga Nasional Berhad (TNB)"
        assert r["pay_full"] is True

    def test_syabas_keyword(self):
        r = _parse_pay_bill_prefill("pay my water bill syabas")
        assert r is not None
        assert r["biller_name"] == "Syabas (Water)"

    def test_unifi_keyword(self):
        r = _parse_pay_bill_prefill("pay unifi broadband bill")
        assert r is not None
        assert r["biller_name"] == "Unifi Broadband"

    def test_custom_amount(self):
        r = _parse_pay_bill_prefill("pay RM 80 to TNB")
        assert r is not None
        assert r["biller_name"] == "Tenaga Nasional Berhad (TNB)"
        assert r["pay_full"] is False
        assert r["amount"] == 80.0

    def test_returns_none_if_no_biller(self):
        assert _parse_pay_bill_prefill("I want to pay a bill") is None

    def test_electricity_alias(self):
        r = _parse_pay_bill_prefill("pay my electricity bill")
        assert r is not None
        assert r["biller_name"] == "Tenaga Nasional Berhad (TNB)"


# ── _detect_emotion + _EMOTION_PROSODY ───────────────────────────────────────
class TestDetectEmotion:
    def test_excited_for_promotion(self):
        assert _detect_emotion("You can earn 2x reward points with our promotion!") == "excited"

    def test_excited_for_cashback(self):
        assert _detect_emotion("Get 5% cashback on petrol purchases.") == "excited"

    def test_empathetic_for_apology(self):
        assert _detect_emotion("I'm sorry, I'm unable to process that request.") == "empathetic"

    def test_empathetic_for_unfortunately(self):
        assert _detect_emotion("Unfortunately your balance is insufficient.") == "empathetic"

    def test_cheerful_for_greeting(self):
        assert _detect_emotion("Hi! I'm happy to help you with that.") == "cheerful"

    def test_cheerful_for_confirmation(self):
        assert _detect_emotion("Great, your transfer is all set!") == "cheerful"

    def test_default_for_transactional(self):
        assert _detect_emotion("Your current account balance is RM 12,480.50.") == "default"

    def test_default_for_generic_question(self):
        assert _detect_emotion("Your loan next due date is 1 May 2026.") == "default"

    def test_prosody_dict_has_all_emotions(self):
        for emotion in ("cheerful", "empathetic", "excited", "default"):
            assert emotion in _EMOTION_PROSODY
            assert "rate" in _EMOTION_PROSODY[emotion]
            assert "pitch" in _EMOTION_PROSODY[emotion]

    def test_prosody_rate_format(self):
        for prosody in _EMOTION_PROSODY.values():
            assert prosody["rate"].endswith("%"), f"rate should be e.g. '+10%', got {prosody['rate']}"

    def test_excited_prosody_faster_than_default(self):
        # Excited should have a higher positive rate than default
        excited_rate = int(_EMOTION_PROSODY["excited"]["rate"].replace("%", "").replace("+", ""))
        default_rate = int(_EMOTION_PROSODY["default"]["rate"].replace("%", "").replace("+", ""))
        assert excited_rate > default_rate

