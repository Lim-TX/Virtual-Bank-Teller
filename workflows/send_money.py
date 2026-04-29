"""Send Money workflow — data helper + Streamlit surface renderer."""
import streamlit as st


def build_send_money_context(account: dict) -> dict:
    """Extract the data needed by the Send Money surface from MOCK_ACCOUNT."""
    sm = account["send_money"]
    return {
        "daily_limit": sm["daily_limit"],
        "per_transaction_limit": sm["per_transaction_limit"],
        "remaining_today": sm["remaining_today"],
        "fee": sm["fee"],
        "transfer_types": sm["transfer_types"],
        "supported_banks": sm["supported_banks"],
        "source_accounts": [
            {"label": f"Current Account  {account['account_number']}", "balance": account["balance"]},
            {"label": f"Savings Account  {account['savings_account']['account_number']}", "balance": account["savings_account"]["balance"]},
        ],
    }


def render_send_money(account: dict) -> None:
    """Render the Send Money workflow inside the dashboard column."""
    ctx = build_send_money_context(account)

    # Apply any pending prefill (set by the chat prefill parser before navigation).
    # Use get() not pop() so clicking the action link multiple times re-applies the same prefill.
    # Cleared when the user sends a new message (overwritten or set to None).
    _pf = st.session_state.get("_sm_prefill", None)
    if _pf:
        if _pf.get("source"):
            st.session_state["sm_source"] = _pf["source"]
        if _pf.get("type"):
            st.session_state["sm_type"] = _pf["type"]
        if _pf.get("recipient"):
            st.session_state["sm_recipient"] = _pf["recipient"]
        if _pf.get("amount") is not None:
            st.session_state["sm_amount"] = float(_pf["amount"])
        if _pf.get("note"):
            st.session_state["sm_note"] = _pf["note"]

    st.markdown('<h3 style="margin-top:16px;">Send Money</h3>', unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Select source account**")
        source_labels = [a["label"] for a in ctx["source_accounts"]]
        selected_source = st.selectbox("From", source_labels, key="sm_source", label_visibility="collapsed")
        for a in ctx["source_accounts"]:
            if a["label"] == selected_source:
                st.caption(f"Available: {a['balance']}")

        st.markdown("**Recipient**")
        transfer_type = st.selectbox("Transfer type", ctx["transfer_types"], key="sm_type")
        recipient = st.text_input("Recipient phone / IC / account number", key="sm_recipient", placeholder="e.g. 0123456789")
        if transfer_type not in ("DuitNow (by phone/IC/account number)",):
            recipient_bank = st.selectbox("Recipient bank", ctx["supported_banks"], key="sm_bank")

        st.markdown("**Amount & Note**")
        amount = st.number_input("Amount (RM)", min_value=1.0, max_value=10000.0, step=1.0, key="sm_amount")
        note = st.text_input("Transfer note (optional)", key="sm_note", placeholder="e.g. Rent for April")

        st.markdown("<br>", unsafe_allow_html=True)
        confirm = st.button("Review Transfer", key="sm_review", type="primary", use_container_width=True)
        if confirm:
            if not recipient:
                st.warning("Please enter a recipient.")
            else:
                st.success(
                    f"**Demo — Transfer not executed.**  \n"
                    f"RM {amount:.2f} to '{recipient}' via {transfer_type}.  \n"
                    f"Note: {note or '—'}"
                )

    with col_right:
        st.markdown(
            f"""
            <div style='background:#F8F8F8;border-radius:12px;padding:20px 18px;font-size:14px;min-height:700px;'>
            <div style='font-weight:600;font-size:15px;margin-bottom:14px;color:#1A1A1A;'>Transfer limits</div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Daily limit</span><br><b>{ctx['daily_limit']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Per-transaction limit</span><br><b>{ctx['per_transaction_limit']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Remaining today</span><br><b>{ctx['remaining_today']}</b></div>
            <div><span style='color:#888'>Fee</span><br><b>{ctx['fee']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
