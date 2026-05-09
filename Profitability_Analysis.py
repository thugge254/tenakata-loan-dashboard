import streamlit as st
import pandas as pd
import plotly.express as px
from utils import calculate_average_interest_rate, calculate_average_loan_size
from utils import TEXT_COLOR

def show_profitability_analysis(df):
    #  a copy 
    df = df.copy()

        # --- KPI Section ---
    # Calculate KPIs
    Average_Interest_Rate = calculate_average_interest_rate(df)
    Average_Loan_Size = calculate_average_loan_size(df)
    Estimated_Interest_Revenue = (df["loan_amount"] * (df["rate_of_interest"] / 100)).sum()
    Average_Upfront_Charges = df["Upfront_charges"].mean()

    # Create  estimated interest revenue per loan column in the data
    df["estimated_interest"] = (
        df["loan_amount"] * df["rate_of_interest"] / 100
)


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
   # Create two columns for the visuals
    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            df,
            x="loan_amount",
            y="estimated_interest",
            color="loan_type",
            color_discrete_sequence=["#E53935","#FFB300","#0B3C49"],
            hover_data=["Credit_Score", "income", "Region"],
            title="Loan Amount vs Interest Revenue"
        )

        fig.update_layout(
            title_font_color=TEXT_COLOR,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---- Clean data BEFORE second chart ----
        df["income"] = pd.to_numeric(df["income"], errors="coerce")
        df["income"] = df["income"].fillna(df["income"].median())

    with col4:
        fig = px.scatter(
            df,
            x="loan_amount",
            y="estimated_interest",
            size="income",
            color="Region",
            color_discrete_sequence=["#E53935", "#0B3C49", "#FFB300"],
            hover_data=["Credit_Score", "loan_type"],
            title="Profitability Bubble Chart"
        )

        fig.update_layout(
            title_font_color=TEXT_COLOR,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)