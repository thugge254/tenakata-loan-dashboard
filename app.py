# -*- coding: utf-8 -*-
import streamlit as st
import os
import pandas as pd
from PIL import Image
import plotly.express as px
from Portfolio_Distribution import show_portfolio_distribution
from Predictive_Insights import show_prediction_analysis
from utils import purpose_map, calculate_total_loan_portfolio, calculate_average_interest_rate
from utils import calculate_average_loan_size, calculate_total_loans_issued
from Risk_Analysis import show_risk_analysis
from  Profitability_Analysis import show_profitability_analysis


st.set_page_config(
    page_title='Tenakata Dashboard',
    page_icon=':chart_with_upwards_trend:',
    layout='wide'
)

if "page" not in st.session_state:
    st.session_state.page = "OVERVIEW"

st.markdown("""
    <style>
        /* Main container */
        .block-container {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* Remove app top spacing */
        .stApp {
            margin-top: 0rem;
            padding-top: 0rem;
        }

        /* Hide Streamlit header */
        header {
            visibility: hidden;
        }

        /* Hide toolbar */
        .stToolbar {
            display: none;
        }

        /* Remove title spacing */
        h1 {
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv("Loan_default.csv")
df.columns = df.columns.str.strip()

image = Image.open("tena-kata.jpg")


col1, col2 = st.columns([0.15, 0.85])

with col1:
    st.image(image, width=120)
    st.markdown(
        "<p style='font-size:12px; color:#0B3C49; margin-top:5px;'>"
        "<b>Source:</b> Kaggle – Loan Default Dataset<br>Latest Available Data"
        "</p>",
        unsafe_allow_html=True
    )

html_title = """
<h1 style='
    text-align:center;
    margin:0;
    padding:0;
    color:#0B3C49;
    line-height:1.2;
    font-size:clamp(1.2rem, 4vw, 2.8rem);
    word-wrap:break-word;
    '>
    Empowering MSMEs Through Data: Tenakata Loan Portfolio Analysis
    </h1>
    """

with col2:
    st.markdown(html_title, unsafe_allow_html=True)

bt1, bt2, bt3, bt4, bt5  = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])

with bt1:
    st.markdown("""
    <style>
    div.stButton > button {
        width: clamp(140px, 60vw, 230px) !important;
        height: clamp(42px, 6vw, 50px) !important;
        font-size: clamp(13px, 2.5vw, 16px) !important;
        font-weight: 700 !important;
        color: white !important;
        background-color: #0B3C49 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #E53935 !important;
        cursor: pointer;
    }

    </style>
    """, unsafe_allow_html=True)

    if st.button("📊 OVERVIEW"):
        st.session_state.page = "OVERVIEW"

with bt2:

    st.markdown("""

    <style>
                
    /* Button base */

    div.stButton > button {

        width: 230px;            

        height: 50px;

        font-size: 16px;

        font-weight: 700;

        color: white;        

        background-color: #0B3C49;

        border-radius: 12px;              

        transition: all 0.3s ease;

    }

    /* Hover effect */

    div.stButton > button:hover {

        color: white;

        background-color: #E53935;

        cursor: pointer;

    }

    /* Clicked / active */

    div.stButton > button:active {

        background-color: #E53935;

        color: white;

    }

    div.stButton > button:focus {

        outline: none !important;

        box-shadow: none !important;

        border: none !important;

    }

    </style>

    """, unsafe_allow_html=True) 


    if st.button("📈 PORTFOLIO DISTRIBUTION"):
        st.session_state.page = "PORTFOLIO"

with bt3:
    st.markdown("""
    <style>

    /* Button base */
    div.stButton > button {
        width: 230px;             
        height: 50px;
        font-size: 16px;
        font-weight: 700;
        color: white;        
        background-color: #0B3C49;
        border-radius: 12px;              
        transition: all 0.3s ease; 
    }

    /* Hover effect */
    div.stButton > button:hover {
        color: white;
        background-color: #E53935;
        cursor: pointer;
    }
                
    /* Clicked / active */
    div.stButton > button:active {
        background-color: #E53935;
        color: white;
    }
    /* Remove default focus outline and glow after button click */
    div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("⚠️ RISK ANALYSIS"):
        st.session_state.page = "RISK"

with bt4:
    st.markdown("""
    <style>

    /* Button base */
    div.stButton > button {
        width: 230px;             
        height: 50px;
        font-size: 16px;
        font-weight: 700;
        color: white;        
        background-color: #0B3C49;
        border-radius: 12px;              
        transition: all 0.3s ease; 
    }

    /* Hover effect */
    div.stButton > button:hover {
        color: white;
        background-color: #E53935;
        cursor: pointer;
    }

    /* Clicked / active */
    div.stButton > button:active {
        background-color: #E53935;
        color: white;
    }
    /* Remove default focus outline and glow after button click */
    div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("💰 PROFITABILITY ANALYSIS"):
        st.session_state.page = "PROFITABILITY"
        
with bt5:
    st.markdown("""
    <style>

    /* Button base */
    div.stButton > button {
        width: 230px;             
        height: 50px;
        font-size: 16px;
        font-weight: 700;
        color: white;        
        background-color: #0B3C49;
        border-radius: 12px;              
        transition: all 0.3s ease; 
    }

    /* Hover effect */
    div.stButton > button:hover {
        color: white;
        background-color: #E53935;
        cursor: pointer;
    }

    /* Clicked / active */
    div.stButton > button:active {
        background-color: #E53935;
        color: white;
    }
    /* Remove default focus outline and glow after button click */
    div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("🔮 PREDICTIVE INSIGHTS"):
         st.session_state.page = "PREDICT"

# --- OVERVIEW PAGE ---
if st.session_state.page == "OVERVIEW":

    # --- KPI Section ---
    # Calculate KPIs
    Total_Loan_Portfolio_Value = calculate_total_loan_portfolio(df)
    Number_of_Active_Loans = len(df)
    Average_Loan_Size = calculate_average_loan_size(df)
    Default_Rate = df["Status"].mean()             
    Average_Credit_Score = df["Credit_Score"].mean() 
    npl_loans = df[df["Status"] == 1].shape[0]
    total_loans = len(df)
    NPL_Ratio = npl_loans/total_loans
    Average_Interest_Rate = calculate_average_interest_rate(df)
    avg_income = df["income"].mean()

    Total_Loans_Issued = calculate_total_loans_issued(df)
    total_loans = df["loan_amount"].sum() 
    unique_borrowers = df["ID"].nunique()

    avg_loans_per_borrower = total_loans / unique_borrowers

    # Transform the column
    df["loan_purpose"] = df["loan_purpose"].map(purpose_map)

    # Display KPIs in five columns
    kpi1, kpi2, kpi3, kpi4, kpi5,  = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    with kpi1:
     st.markdown(
        f"""
        <div style="font-size:18px; font-weight:bold; color:#0B3C49">💰 Total Loan Portfolio </div>
        <div style="font-size:26px; font-weight:900; color:#E53935;">{Total_Loan_Portfolio_Value:,.0f}</div>
        <div style="font-size:14px; color:gray;">
        Total value of loans issued to borrowers within the portfolio.
        Represents the oveall exposure.
        </div>
        """,
        unsafe_allow_html=True
    )
    with kpi2:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🟢 Number of Active Loans</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Number_of_Active_Loans:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            Total number of loans that are currently active (approved and disbursed loans).
            </div>
          """,
            unsafe_allow_html=True
        )
    with kpi3:
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

    with kpi4:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🚨 Default Rate</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Default_Rate:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            The proportion of loans that have defaulted compared to the total number of loans.
            </div>
             """,
            unsafe_allow_html=True
        )
    
    with kpi5:
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

    with kpi6:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">⚠️ NPL Ratio</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{NPL_Ratio:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            The proportion of non-performing loans compared to the total loan portfolio.
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi7:
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

        with kpi8:
            st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">💵 Average Income</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{avg_income:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            The average income of borrowers in the portfolio. It reflects the typical earning capacity of clients.
            </div>
            """,
            unsafe_allow_html=True
        )

        with kpi9:
            st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🧾 Total Loans Issued</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{Total_Loans_Issued:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            Total number of loans that are currently active (approved and disbursed loans).
            </div>
          """,
            unsafe_allow_html=True
        )

        with kpi10:
            st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">📦 Average Loan Amount</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{avg_loans_per_borrower:,.0f}</div>
            <div style="font-size:14px; color:gray;">
            The average number of loans taken per borrower,(total loans/divided by the number of unique borrowers.
            </div>
            """,
            unsafe_allow_html=True
     )

# --- PORTFOLIO DISTRIBUTION ---
elif st.session_state.page == "PORTFOLIO":
    show_portfolio_distribution(df, purpose_map)

# -- RISK ANALYSIS --
if st.session_state.page == "RISK":
    show_risk_analysis(df)

if st.session_state.page == "PROFITABILITY":
    show_profitability_analysis(df)
if st.session_state.page == "PREDICT":
    show_prediction_analysis(df)
