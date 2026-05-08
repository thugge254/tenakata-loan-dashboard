import streamlit as st
import plotly.express as px
from utils import status_map, calculate_total_loan_portfolio
from utils import calculate_average_loan_size, calculate_total_loans_issued


def show_portfolio_distribution(df, purpose_map):
        #  a copy 
        df = df.copy()

        # Apply mapping safely
        df["loan_purpose"] = df["loan_purpose"].map(purpose_map)

        # compute kpis
        Total_Loan_Portfolio_Value = calculate_total_loan_portfolio(df)
        Average_Loan_Size = calculate_average_loan_size(df)
        Total_Loans_Issued = calculate_total_loans_issued(df)

        kpi10_5, kpi11_5, kpi12_5 = st.columns([0.33, 0.33, 0.33])
        kpi10, kpi11, kpi12 = st.columns([0.33, 0.33, 0.33])

        with kpi10_5:
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
            
        with kpi11_5:
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
            
        with kpi12_5:
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
            # Map status labels
            df["Status_Label"] = df["Status"].map(status_map)

            # Count distribution
            status_counts = df["Status_Label"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            # Convert to percentage (optional but recommended)
            status_counts["Percent"] = (
                status_counts["Count"] / status_counts["Count"].sum() * 100
            )

            # 📊 Plot
            fig = px.bar(
                status_counts,
                x="Status",
                y="Percent",
                text="Percent",
                title="📊 Loan Status Distribution",
                color="Status",
                color_discrete_map={
                    "No Default": "#0B3C49",
                    "Default": "#E53935"
                    }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                width=0.6,
                textfont=dict(size=14),
                cliponaxis=False
            )

            fig.update_layout(
                title_x=0,
                yaxis_title="Percentage (%)",
                showlegend=False,
                margin=dict(t=60)
            )
            st.plotly_chart(fig, use_container_width=True)

        with kpi11:
            df_gender = df[df["Gender"] != "Sex Not Available"]

            gender_default = df_gender.groupby("Gender")["Status"].mean().reset_index()
            gender_default.columns = ["Gender", "Default_Rate"]

            fig = px.bar(
                gender_default,
                x="Gender",
                y="Default_Rate",
                text="Default_Rate",
                title="🚨 Default Rate by Gender",
                color="Gender",
                color_discrete_map={
                    "Male": "#0B3C49",
                    "Female": "#E53935",
                    "Joint": "#0068C9"
                    }
            )

            fig.update_traces(
                texttemplate='%{text:.1%}',
                textposition="outside",
                        textfont=dict(
                        color="#0B3C49",  
                        size=14,           
                        family="bold" 
                    ),
                cliponaxis=False
            )
            fig.update_layout(
                yaxis_title="Default Rate %"
                )
            st.plotly_chart(fig, use_container_width=True)

        with kpi12:
            # Transform the column
            df["Status_label"] = df["Status"].map(status_map)

            # Compute percentages
            loan_purpose_counts1 = df["loan_purpose"].value_counts().reset_index()
            loan_purpose_counts1.columns = ["Loan Purpose", "Count"]

            loan_purpose_counts1["Percent"] = (
            loan_purpose_counts1["Count"] / loan_purpose_counts1["Count"].sum() * 100
            )

            # 📊 Chart
            fig = px.bar(
                loan_purpose_counts1,
                x="Loan Purpose",
                y="Percent",
                title="📊 Loan Distribution by Purpose",
                text="Percent",
                color="Loan Purpose",
                color_discrete_map={
                    "Business": "#0B3C49",
                    "Property": "#1f77b4",
                    "Consumption": "#0B3C49",
                    "Education": "#E53935"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                width=0.8,
                textfont=dict(
                    color="#0B3C49",
                    size=14,
                    family="bold"
                ),
                cliponaxis=False
            )

            fig.update_layout(
                margin=dict(t=60),
                bargap=0.05,
                yaxis_title="Loan Purpose %",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)







