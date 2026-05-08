import streamlit as st
import plotly.express as px
from utils import calculate_average_interest_rate, calculate_average_loan_size

def show_profitability_analysis(df):
    #  a copy 
    df = df.copy()

        # --- KPI Section ---
    # Calculate KPIs
    Average_Interest_Rate = calculate_average_interest_rate(df)
    Average_Loan_Size = calculate_average_loan_size(df)
    Estimated_Interest_Revenue = (df["loan_amount"] * (df["rate_of_interest"] / 100)).sum()
    Average_Upfront_Charges = df["Upfront_charges"].mean()


    # Display KPIs in four columns
    kpi18, kpi19, kpi20, kpi21 = st.columns([0.25, 0.25, 0.25, 0.25])

    with kpi18:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🧮 Average Interest Rate</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Average_Interest_Rate:,.2f}%</div>
            <div style="font-size:14px; color:gray;">
            Average interest rate applied across all loans in the portfolio.
            Reflects the cost of borrowing for clients.
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi19:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">💰 Average Loan Size</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Average_Loan_Size:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            The average amount issued per loan, calculated as total loan portfolio/number of active loans.
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi20:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">📈 Estimated Interest Revenue</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Estimated_Interest_Revenue:,.0f}</div>

            <div style="font-size:14px; color:gray;">
            Estimated total interest income generated from all loans in the portfolio
            based on loan amounts and interest rates.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi21:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🧾 Average Upfront Charges</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Average_Upfront_Charges:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            The average upfront fees charged to borrowers during loan issuance.
            </div>
            """,
            unsafe_allow_html=True
        )