import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import plotly.graph_objects as go



def show_prediction_analysis(df):
    
    # 1. Create a copy to prevent modifying the original dataframe
    df = df.copy()

    # Display KPIs in five columns
    kpi22, kpi23, kpi24, kpi25, kpi26 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])

    # --- KPI Section ---
    # Calculate KPIs
    Default_Rate = df["Status"].mean()


      # Display KPIs in five columns
    with kpi22:
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
    



    categorical_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    ######## Define features and target variable
    # leakage features that should be excluded from the model to prevent data leakage

    leakage_cols = [
    "Interest_rate_spread",
    "rate_of_interest",
    "Upfront_charges",
    "ID"
]

    X = df.drop(columns=[
    "Status",
    "Interest_rate_spread",
    "rate_of_interest",
    "Upfront_charges",
    "ID"
    ])
    y = df["Status"]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )


    # Train an XGBoost model
    model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=2,
    min_child_weight=5,
    random_state=42,
    eval_metric="logloss"
    )

    model.fit(X_train, y_train)


    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print(classification_report(y_test, y_pred))


    # Save model
    joblib.dump(model, "xgb_model.pkl")

    # Compute risk probabilities
    risk_scores = model.predict_proba(X)[:, 1]

    # Compute average risk score
    average_risk_score = risk_scores.mean() * 100

    with kpi23:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">📊 Average Risk Score</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{average_risk_score:.2f}%</div>
            <div style="font-size:14px; color:gray;">
            The average predicted probability of default across all borrowers in the portfolio.
            </div>
            """,
            unsafe_allow_html=True
        )
    # calculate risk probabilities values 
    risk_scores = model.predict_proba(X)[:, 1]

    # Define “High Risk” threshold
    threshold = 0.7

    # Count high-risk borrowers
    high_risk_borrowers = (risk_scores >= threshold).sum()


    with kpi24:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">⚠️ High-Risk Borrowers</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{high_risk_borrowers:,}</div>
            <div style="font-size:14px; color:gray;">
            Number of borrowers with a predicted default probability above 70%, termed as “High Risk”.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    with kpi25:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">🎯 Model Accuracy</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{accuracy:.0f}%</div>
            <div style="font-size:14px; color:gray;">
            Percentage of correctly classified loan outcomes based on the XGBoost prediction model. 
            </div>
            """,
            unsafe_allow_html=True
        )

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute AUC
    auc = roc_auc_score(y_test, y_prob) * 100

    with kpi26:
        st.markdown(
            f"""
            <div style="font-size:18px; font-weight:bold; color:#0B3C49">📈 AUC</div>
            <div style="font-size:26px; font-weight:900; color:#E53935;">{auc:.0f}%</div>
            <div style="font-size:14px; color:gray;">
            Area Under the ROC Curve for the XGBoost prediction model.
            </div>
            """,
            unsafe_allow_html=True
        )
    importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print(importance.head(10))


    col1, col2, col3 = st.columns([0.33, 0.33, 0.33])

    X = df.drop(columns=[
    "Status",
    "Interest_rate_spread",
    "rate_of_interest",
    "Upfront_charges",
    "ID"
    ])

    # Predict probabilities
    risk_scores = model.predict_proba(X)[:, 1]

    # Portfolio average risk
    portfolio_risk = risk_scores.mean() * 100

    with col1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=portfolio_risk,

            title={
                'text': "Risk Measure",
                'font': {
                            'size': 22,
                            'color': '#0B3C49',
                            'family': "Arial Black"
                     }
            },

            number={
            'suffix': "%",
            'font': {
                'size': 50,
                'color': '#0B3C49', 
                'family': "Arial Black"
            }
        },

            gauge={
                'axis': {
                'range': [0, 100],
                'tickwidth': 2,       
                'tickcolor': "black",
                'tickfont': {
                'family': "Arial Black",
                'size': 14,             
                'color': "black"        
                }
                },

                'bar': {'color': "#E53935"},

                'steps': [
                    {'range': [0, 30], 'color': "#07f041"},
                    {'range': [30, 70], 'color': "#f0cd07"},
                    {'range': [70, 100], 'color': "#eb4034"}
                ],

                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': portfolio_risk
                }
            }
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=100, b=20), # Increased top margin (t)
            title={
                'text': "<b>Portfolio Risk Distribution</b>",
                'y': 0.85, 
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': '#0B3C49', 'family': "Arial Black"}
            }
            )

        st.plotly_chart(fig, use_container_width=True)

   # --- COLUMN 2: BAR CHART ---
    with col2:
        # Creating the Risk Level segments
        df['Risk_Level'] = pd.cut(
            risk_scores,
            bins=[0, 0.30, 0.70, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        # Count each category
        risk_counts = df['Risk_Level'].value_counts().reindex(
            ['Low Risk', 'Medium Risk', 'High Risk']
        ).reset_index()

        risk_counts.columns = ['Risk_Level', 'Count']

        # Convert to percentage
        total = risk_counts['Count'].sum()
        risk_counts['Percentage'] = (risk_counts['Count'] / total * 100).round(1)


        fig_bar = px.bar(
            risk_counts,
            x='Risk_Level',
            y='Count',
            title="<b>Risk Segmentation</b>",
            color='Risk_Level',
            color_discrete_map={
                'Low Risk': '#00E676',   
                'Medium Risk': '#FFD600',
                'High Risk': '#FF1744'
                },
              text='Percentage'   
        )

        # Force labels outside the bars
        fig_bar.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            cliponaxis=False
        )
        fig_bar.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=100, b=20),
            title={
                'text': "<b>Risk Segmentation</b>",
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': '#0B3C49', 'family': "Arial Black"}
            },
            yaxis_title="Number of Clients",
            xaxis_title="Risk Level",
            showlegend=False,
            plot_bgcolor="white"

        )

        st.plotly_chart(fig_bar, use_container_width=True)

        with col3:
            # --- SAFETY CHECK ---
            if 'prediction_probability' not in df.columns:

                feature_cols = [
                    'year', 'loan_limit', 'Gender', 'approv_in_adv', 'loan_type',
                    'loan_purpose', 'Credit_Worthiness', 'open_credit',
                    'business_or_commercial', 'loan_amount', 'term',
                    'Neg_ammortization', 'interest_only', 'lump_sum_payment',
                    'property_value', 'construction_type', 'occupancy_type',
                    'Secured_by', 'total_units', 'income', 'credit_type',
                    'Credit_Score', 'co-applicant_credit_type', 'age',
                    'submission_of_application', 'LTV', 'Region',
                    'Security_Type', 'dtir1'
                ]

                X_safe = df[feature_cols]

                df['prediction_probability'] = model.predict_proba(X_safe)[:, 1]

            # ------------------------------------------------------------------
            # Aggregate risk
            # ------------------------------------------------------------------
            heatmap_data = (
                df.groupby(['Region', 'loan_type'])['prediction_probability']
                .mean()
                .reset_index()
            )

            # ------------------------------------------------------------------
            # Pivot table
            # ------------------------------------------------------------------
            heatmap_pivot = heatmap_data.pivot(
                index='Region',
                columns='loan_type',
                values='prediction_probability'
            )

            # ------------------------------------------------------------------
            # Heatmap
            # ------------------------------------------------------------------
            fig_heat = px.imshow(
                heatmap_pivot,
                labels=dict(
                    x="Loan Type",
                    y="Region",
                    color="Avg Risk Probability"
                ),
                color_continuous_scale='RdYlGn_r',
                text_auto=".2f"
            )

            # ------------------------------------------------------------------
            # Layout
            # ------------------------------------------------------------------
            fig_heat.update_layout(
                height=400,

                # THIS FIXES THE TITLE OVERLAP
                margin=dict(l=30, r=30, t=100, b=20),

                title={
                    'text': "<b>Risk Heatmap</b>",
                    'y': 0.85,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {
                        'size': 22,
                        'color': '#0B3C49',
                        'family': "Arial Black"
                    }
                },

                xaxis_title="Loan Category",
                yaxis_title="Geographic Region",

                plot_bgcolor="white",

                font=dict(size=10)
            )

            # Rotate labels
            fig_heat.update_xaxes(tickangle=-45)

            # ------------------------------------------------------------------
            # Display
            # ------------------------------------------------------------------
            st.plotly_chart(fig_heat, use_container_width=True)