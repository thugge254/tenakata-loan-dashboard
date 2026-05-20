import streamlit as st
import pandas as pd
import plotly.express as px


def show_risk_analysis(df):
    

    # Display KPIs in columns
    kpi13, kpi14, kpi15, kpi16, kpi17 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])

    # --- KPI Section ---
    Average_Credit_Score = df["Credit_Score"].mean(skipna=True) 
    average_dti = df["dtir1"].mean(skipna=True)
    average_ltv = df["LTV"].mean(skipna=True)
    interest_only_count = (df["interest_only"] == "int_only").sum()
    total_loans = len(df)
    percent_interest_only = (interest_only_count / total_loans) * 100 if total_loans > 0 else 0

    with kpi13:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🏦 Average Credit Score</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Average_Credit_Score:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            The average credit score of borrowers in the portfolio, used to assess overall borrower creditworthiness and risk.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Data Cleaning & Risk Logic ---
    numeric_cols = ["Credit_Score", "dtir1", "LTV"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean strings safely using astype(str) to handle any NaN values
    df["interest_only"] = df["interest_only"].astype(str).str.strip().str.lower()
    df["Neg_ammortization"] = df["Neg_ammortization"].astype(str).str.strip().str.lower()

    # -- Create a Risk Flag Column --
    # Flagging criteria: Low credit, high debt, high loan-to-value, or risky loan types
    df["risk_flag"] = (
        (df["Credit_Score"].fillna(0) < 600) |
        (df["dtir1"].fillna(0) > 40) |
        (df["LTV"].fillna(0) > 90) |
        (df["interest_only"] == "int_only") |
        (df["Neg_ammortization"] == "neg_amm")
    )

    
    # 1. Convert the boolean column to 0 and 1
    df["risk_flag"] = df["risk_flag"].astype(int)

    total_high_risk = int(df["risk_flag"].sum())
    with kpi14:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">⚠️ Total High Risk Loans</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{total_high_risk:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            Total number of accounts considered as High Risk based on risk flagging criteria.
            </div>
             """,
            unsafe_allow_html=True
        )

    with kpi15:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">⚖️ Average DTI Ratio</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{average_dti:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            Shows how much debt a borrower have compared to their income. Higher values mean higher borrowing risk.
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi16:  
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🏠 Average LTV</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{average_ltv:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            Loan-to-Value ratio. A higher percentage indicates less borrower equity and higher lender risk.
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi17: # Or next available KPI column
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">💳 % Interest-Only</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{percent_interest_only:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            A proportion of loans in the portfolio with interest-only payment structures as per the Data.
            </div>
            """,
            unsafe_allow_html=True
        )
    # Create two columns for the visuals
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            df, 
            x="Credit_Score", 
            nbins=40,
            title="📊 Credit Score Distribution Histogram",
            color_discrete_sequence=["#0B3C49"]
        )
        
        fig_hist.update_layout(
            xaxis_title="Credit Score",
            title_x=0,
            yaxis_title="Number of Borrowers",
            bargap=0.1,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_box = px.box(
            df, 
            x="Credit_Score",
            title="📦 A box plot of Credit Score",
            points="outliers",
            color_discrete_sequence=["#E53935"] # Highlighting spread in red
        )
        
        fig_box.update_layout(
            xaxis_title="Credit Score",
            title_x=0,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_box, use_container_width=True)