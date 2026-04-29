"""Pay Loan / Financing workflow — data helper + Streamlit surface renderer."""
import streamlit as st


def build_pay_loan_context(account: dict) -> dict:
    """Extract the data needed by the Pay Loan surface from MOCK_ACCOUNT."""
    loan = account["loan"]
    pl = account["pay_loan"]
    return {
        "loan_type": loan["type"],
        "outstanding_balance": loan["outstanding_balance"],
        "monthly_instalment": loan["monthly_instalment"],
        "next_due_date": loan["next_due_date"],
        "remaining_tenure": loan["remaining_tenure"],
        "payment_options": pl["payment_options"],
        "payment_source": pl["payment_source"],
        "early_settlement_fee": pl["early_settlement_fee"],
    }


def render_pay_loan(account: dict) -> None:
    """Render the Pay Loan / Financing workflow inside the dashboard column."""
    ctx = build_pay_loan_context(account)

    # Apply any pending prefill from chat parser
    _pf = st.session_state.get("_pl_prefill")
    if _pf:
        if _pf.get("option"):
            st.session_state["pl_option"] = _pf["option"]
        if _pf.get("partial") is not None:
            st.session_state["pl_partial"] = float(_pf["partial"])

    st.markdown('<h3 style="margin-top:16px;">Pay Loan / Financing</h3>', unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            f"""
            <div style='background:#FFF8F0;border-radius:12px;padding:16px 18px;margin-bottom:16px;font-size:14px;'>
            <div style='color:#888;margin-bottom:4px;'>{ctx['loan_type']}</div>
            <div style='margin-top:6px;'><span style='color:#888'>Outstanding balance</span><br>
            <b style='font-size:22px;color:#C8102E;'>{ctx['outstanding_balance']}</b></div>
            <div style='margin-top:8px;display:flex;gap:24px;'>
            <div><span style='color:#888'>Monthly instalment</span><br><b>{ctx['monthly_instalment']}</b></div>
            <div><span style='color:#888'>Next due</span><br><b>{ctx['next_due_date']}</b></div>
            <div><span style='color:#888'>Remaining tenure</span><br><b>{ctx['remaining_tenure']}</b></div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Payment option**")
        option_labels = [f"{o['option']}  —  {o['amount']}" for o in ctx["payment_options"]]
        selected_option = st.radio("Payment type", option_labels, key="pl_option", label_visibility="collapsed")

        partial_amount = None
        if "Partial" in selected_option:
            partial_amount = st.number_input("Partial payment amount (RM)", min_value=500.0, step=100.0, key="pl_partial")

        st.markdown(f"**Payment source:** {ctx['payment_source']}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Confirm Payment", key="pl_confirm", type="primary", use_container_width=True):
            amount_str = f"RM {partial_amount:.2f}" if partial_amount else selected_option.split("—")[-1].strip()
            st.success(f"**Demo — Payment not processed.**  \nScheduled: {amount_str} from {ctx['payment_source']}.")

    with col_right:
        st.markdown(
            f"""
            <div style='background:#F8F8F8;border-radius:12px;padding:20px 18px;font-size:14px;min-height:700px;'>
            <div style='font-weight:600;font-size:15px;margin-bottom:14px;color:#1A1A1A;'>Loan summary</div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Loan type</span><br><b>{ctx['loan_type']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Next due date</span><br><b>{ctx['next_due_date']}</b></div>
            <div><span style='color:#888'>Early settlement fee</span><br><b>{ctx['early_settlement_fee']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
