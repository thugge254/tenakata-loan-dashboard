import streamlit as st

def show_risk_analysis(df):
    
    #  a copy 
    df = df.copy()

    # --- KPI Section ---
    Average_Credit_Score = df["Credit_Score"].mean() 


    # Display KPIs in four columns
    kpi13, kpi14, kpi15 = st.columns([0.33, 0.33, 0.33])

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
