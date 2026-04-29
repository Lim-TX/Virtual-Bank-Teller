"""Pay Card workflow — data helper + Streamlit surface renderer."""
import streamlit as st


def build_pay_card_context(account: dict) -> dict:
    """Extract the data needed by the Pay Card surface from MOCK_ACCOUNT."""
    cc = account["credit_card"]
    pc = account["pay_card"]
    return {
        "card_number": cc["number"],
        "outstanding_balance": cc["outstanding_balance"],
        "minimum_payment": cc["minimum_payment"],
        "due_date": cc["due_date"],
        "payment_options": pc["payment_options"],
        "payment_source": pc["payment_source"],
    }


def render_pay_card(account: dict) -> None:
    """Render the Pay Card workflow inside the dashboard column."""
    ctx = build_pay_card_context(account)

    # Apply any pending prefill from chat parser
    _pf = st.session_state.get("_pc_prefill")
    if _pf:
        if _pf.get("option"):
            st.session_state["pc_option"] = _pf["option"]
        if _pf.get("custom") is not None:
            st.session_state["pc_custom"] = float(_pf["custom"])

    st.markdown('<h3 style="margin-top:16px;">Pay Card</h3>', unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            f"""
            <div style='background:#FFF0F2;border-radius:12px;padding:16px 18px;margin-bottom:16px;font-size:14px;'>
            <div style='color:#888;margin-bottom:4px;'>Credit Card</div>
            <div style='font-weight:700;font-size:17px;color:#1A1A1A;'>{ctx['card_number']}</div>
            <div style='margin-top:10px;'><span style='color:#888'>Outstanding balance</span><br>
            <b style='font-size:22px;color:#C8102E;'>{ctx['outstanding_balance']}</b></div>
            <div style='margin-top:8px;'><span style='color:#888'>Minimum payment</span>&nbsp;
            <b>{ctx['minimum_payment']}</b>&nbsp;&nbsp;<span style='color:#888'>Due</span>&nbsp;<b>{ctx['due_date']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Payment option**")
        option_labels = [f"{o['option']}  —  {o['amount']}" for o in ctx["payment_options"]]
        selected_option = st.radio("Payment amount", option_labels, key="pc_option", label_visibility="collapsed")

        custom_amount = None
        if "Custom" in selected_option:
            custom_amount = st.number_input("Enter custom amount (RM)", min_value=50.0, max_value=1250.0, step=10.0, key="pc_custom")

        st.markdown(f"**Payment source:** {ctx['payment_source']}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Confirm Payment", key="pc_confirm", type="primary", use_container_width=True):
            amount_str = f"RM {custom_amount:.2f}" if custom_amount else selected_option.split("—")[-1].strip()
            st.success(f"**Demo — Payment not processed.**  \nScheduled: {amount_str} from {ctx['payment_source']}.")

    with col_right:
        st.markdown(
            f"""
            <div style='background:#F8F8F8;border-radius:12px;padding:20px 18px;font-size:14px;min-height:700px;'>
            <div style='font-weight:600;font-size:15px;margin-bottom:14px;color:#1A1A1A;'>Payment summary</div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Card</span><br><b>{ctx['card_number']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Selected option</span><br>
            <b>{"Custom amount" if custom_amount else selected_option.split("—")[0].strip()}</b></div>
            <div><span style='color:#888'>Due date</span><br><b>{ctx['due_date']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
