"""Pay Bill workflow — data helper + Streamlit surface renderer."""
import streamlit as st


def build_pay_bill_context(account: dict) -> dict:
    """Extract the data needed by the Pay Bill surface from MOCK_ACCOUNT."""
    pb = account["pay_bill"]
    return {
        "registered_billers": pb["registered_billers"],
        "payment_source": pb["payment_source"],
        "processing_time": pb.get("processing_time", "Processed instantly"),
    }


def render_pay_bill(account: dict) -> None:
    """Render the Pay Bill workflow inside the dashboard column."""
    ctx = build_pay_bill_context(account)

    # Apply any pending prefill from chat parser
    _pf = st.session_state.get("_pb_prefill")
    if _pf:
        biller_name = _pf.get("biller_name")
        if biller_name:
            # Build the exact radio label from account data
            for b in account["pay_bill"]["registered_billers"]:
                if b["biller"] == biller_name:
                    st.session_state["pb_biller"] = (
                        f"{b['biller']}  (due {b['due']}  ·  {b['amount_due']})"
                    )
                    break
        if _pf.get("pay_full") is False and _pf.get("amount") is not None:
            st.session_state["pb_full"] = False
            st.session_state["pb_custom"] = float(_pf["amount"])
        # pay_full=True is the checkbox default — no need to set it explicitly

    st.markdown('<h3 style="margin-top:16px;">Pay Bill</h3>', unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Registered billers**")
        biller_labels = [f"{b['biller']}  (due {b['due']}  ·  {b['amount_due']})" for b in ctx["registered_billers"]]
        selected_biller_label = st.radio("Select biller", biller_labels, key="pb_biller", label_visibility="collapsed")

        selected_biller = next(
            b for b in ctx["registered_billers"]
            if b["biller"] in selected_biller_label
        )

        st.markdown(f"**Amount due:** `{selected_biller['amount_due']}`")
        st.markdown(f"**Ref:** `{selected_biller['account_ref']}`")

        pay_full = st.checkbox("Pay full amount due", value=True, key="pb_full")
        custom_bill_amount = None
        if not pay_full:
            custom_bill_amount = st.number_input("Custom amount (RM)", min_value=1.0, step=1.0, key="pb_custom")

        st.markdown(f"**Payment source:** {ctx['payment_source']}")

        st.markdown("<br>", unsafe_allow_html=True)
        amount_str = selected_biller["amount_due"] if pay_full else (f"RM {custom_bill_amount:.2f}" if custom_bill_amount else "—")
        if st.button("Confirm Payment", key="pb_confirm", type="primary", use_container_width=True):
            st.success(f"**Demo — Payment not processed.**  \n{amount_str} to {selected_biller['biller']} ({selected_biller['account_ref']}).")

    with col_right:
        amount_str = selected_biller["amount_due"] if pay_full else (f"RM {custom_bill_amount:.2f}" if custom_bill_amount else "—")
        st.markdown(
            f"""
            <div style='background:#F8F8F8;border-radius:12px;padding:20px 18px;font-size:14px;min-height:700px;'>
            <div style='font-weight:600;font-size:15px;margin-bottom:14px;color:#1A1A1A;'>Payment summary</div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Biller</span><br><b>{selected_biller['biller']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Ref number</span><br><b>{selected_biller['account_ref']}</b></div>
            <div style='margin-bottom:10px;'><span style='color:#888'>Amount</span><br><b style='font-size:18px;color:#C8102E;'>{amount_str}</b></div>
            <div><span style='color:#888'>Processing</span><br><b>{ctx['processing_time']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
