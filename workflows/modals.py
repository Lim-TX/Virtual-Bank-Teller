"""
Modal surface renderers using st.dialog.

Each function is decorated with @st.dialog and called by the unified
launcher in app.py. Because st.dialog is a decorator, each dialog
must be a separate top-level function.
"""
import streamlit as st


@st.dialog("Promotions")
def show_promotions(promotions: list[dict]) -> None:
    """Display promotions detail modal."""
    for promo in promotions:
        st.markdown(f"#### {promo['title']}")
        st.markdown(promo["description"])
        st.caption(f"Valid until: {promo['valid_until']}")
        st.markdown("---")


@st.dialog("Mailbox")
def show_mailbox(transactions: list[dict]) -> None:
    """Display recent transaction mailbox modal."""
    st.markdown("**Recent Transactions**")
    for txn in transactions:
        cols = st.columns([2, 3, 1])
        cols[0].caption(txn["date"])
        cols[1].markdown(txn["description"])
        color = "#C8102E" if txn["amount"].startswith("-") else "#16A34A"
        cols[2].markdown(
            f"<span style='color:{color};font-weight:700;font-size:14px;'>{txn['amount']}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("")
    st.caption("Showing last 5 transactions. Visit full statement for complete history.")


@st.dialog("Account Summary")
def show_account_summary(account: dict) -> None:
    """Display account summary detail modal."""
    st.markdown("**Current Account**")
    st.markdown(
        f"""
        <div style='background:#F8F8F8;border-radius:10px;padding:14px 18px;font-size:14px;margin-bottom:12px;'>
        <div><span style='color:#888'>Account number</span><br><b>{account['account_number']}</b></div>
        <div style='margin-top:8px;'><span style='color:#888'>Balance</span><br>
        <b style='font-size:22px;color:#C8102E;'>{account['balance']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Savings Account**")
    sa = account["savings_account"]
    st.markdown(
        f"""
        <div style='background:#F8F8F8;border-radius:10px;padding:14px 18px;font-size:14px;'>
        <div><span style='color:#888'>Account number</span><br><b>{sa['account_number']}</b></div>
        <div style='margin-top:8px;'><span style='color:#888'>Balance</span><br>
        <b style='font-size:22px;'>{sa['balance']}</b></div>
        <div style='margin-top:8px;'><span style='color:#888'>Interest rate</span><br><b>{sa['interest_rate']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("System Maintenance Notice")
def show_maintenance(notice: str) -> None:
    """Display system maintenance notice modal."""
    st.warning(notice)
    st.markdown(
        "We apologise for any inconvenience. Please plan your transactions in advance "
        "and ensure all pending payments are completed before the maintenance window."
    )
