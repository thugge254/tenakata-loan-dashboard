import joblib
import numpy as np
import streamlit.components.v1 as components
import plotly.io as pio
import textwrap
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline




def show_prediction_analysis(df):
    

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
    

    # identify categorical columns 
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(exclude=["object"]).columns



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

  
    X = df.drop(columns=[
    "Status",
    "Interest_rate_spread",
    "rate_of_interest",
    "Upfront_charges",
    "ID"
    ])

    y = df["Status"]

    # CLEAN column names
    X.columns = X.columns.str.strip()
    
    # define column groups from X (NOT df)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Build preprocessor AFTER split-safe columns
    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
    )

    # Split data again to ensure preprocessor is built on training data only
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    # Build pipeline
    pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
    ])

    # Train the modelon the training data only
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print(classification_report(y_test, y_pred))


    # Save model
    joblib.dump(pipeline, "xgb_model.pkl")

    # Compute risk probabilities
    risk_scores = pipeline.predict_proba(X)[:, 1]

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
    risk_scores = pipeline.predict_proba(X)[:, 1]

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
    y_pred = pipeline.predict(X_test)

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
    y_prob = pipeline.predict_proba(X_test)[:, 1]

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
    # get pipeline components
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # get encoded feature names
    feature_names = preprocessor.get_feature_names_out()

    # get importances
    importances = model.feature_importances_

    # create a DataFrame 
    importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    col_PRD, col_BC, col_HM= st.columns([0.33, 0.33, 0.33])

    X = df.drop(columns=[
    "Status",
    "Interest_rate_spread",
    "rate_of_interest",
    "Upfront_charges",
    "ID"
    ])

    # Predict probabilities
    risk_scores = pipeline.predict_proba(X)[:, 1]

    # Portfolio average risk
    portfolio_risk = risk_scores.mean() * 100

    
    with col_PRD:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=portfolio_risk,
            title={
                'font': {
                    'size': 22,
                    'color': '#0B3C49',
                    'family': "Arial Black"
                }
            },
            number={
                'suffix': "%",
                'font': {
                    'size': 32, 
                    'color': '#0B3C49', 
                    'family': "Arial-Bold"
                }
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                    'tickwidth': 2,       
                    'tickcolor': "black",
                    'tickfont': {
                        'family': "Arial Black",
                        'size': 14,            
                        'color': '#0B3C49'       
                    }
                },
                'bar': {'color': "#0B3C49"},
                'steps': [
                    {'range': [0, 30], 'color': "#07f041"},
                    {'range': [30, 70], 'color': "#f0cd07"},
                    {'range': [70, 100], 'color': "#eb4034"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.70,
                    'value': portfolio_risk
                }
            }
        ))

        fig.update_layout(
            height=220, 
            margin=dict(l=40, r=40, t=10, b=10), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',  
        )

        
        # 2. Convert the figure directly into raw HTML components
        # CHANGE: Named this 'raw_gauge_html' to avoid overwriting the variable name in the f-string loop below
        raw_gauge_html = pio.to_html(fig, include_plotlyjs='cdn', config={"displayModeBar": False})

        # 3. Create a clean HTML layout string
        gauge_card_html = f"""
            <div style="
                margin-top: 42px; /* CHANGE: Added 42px top margin so it perfectly matches the height alignment of your bar chart and heatmap */
                background-color: #ffffff; 
                padding: 15px; 
                height: 270px; 
                border-radius: 20px; 
                border: 1px solid #e6e9ef;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
                font-family: 'Arial Black', Arial, sans-serif;
                box-sizing: border-box; 
                overflow: hidden; 
                ">
                <h3 style="
                    font-size: 18px; 
                    color: #0B3C49; 
                    margin: 10px 0 15px 0; 
                    text-align: center;
                    background-color: transparent;
                ">Portfolio Risk Distribution</h3> 
                
                <div style="margin-top: 10px;"> 
                    {raw_gauge_html} </div>
                
            </div>
        """

        # 4. Render the unified HTML code safely using components.html
        # CHANGE: Increased viewport height to 330 to comfortably clear the new 42px alignment offset spacer
        components.html(gauge_card_html, height=330, scrolling=False)


 

    with col_BC:
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
            height=200,
            # CHANGE: Adjusted padding (t=20, b=30) to give category text room to sit cleanly
            margin=dict(l=40, r=30, t=20, b=30),
            
            # CHANGE: Set axis titles to None to completely disable them and prevent stacking overlaps
            yaxis_title=None,
            xaxis_title=None,
            
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor="white"
        )

          
        # 2. Convert the figure directly into raw HTML components
        BC_html = pio.to_html(fig_bar, include_plotlyjs='cdn', config={"displayModeBar": False})

        # 3. Create a clean HTML layout string
        bar_chart_html = f"""
            <div style="
                margin-top: 42px; 
                background-color: #ffffff; 
                padding: 15px; 
                height: 270px; 
                border-radius: 20px; 
                border: 1px solid #e6e9ef;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
                font-family: 'Arial Black', Arial, sans-serif;
                box-sizing: border-box; 
                overflow: hidden; 
                ">
                <h3 style="
                    font-size: 18px; 
                    color: #0B3C49; 
                    margin: 10px 0 5px 0; /* CHANGE: Reduced bottom margin slightly to lift chart space */
                    text-align: center;
                    background-color: transparent;
                ">Risk Segmentation</h3> 
                
                <div style="margin-top: 5px;">
                    {BC_html}
                </div>
                
            </div>
        """

        # 4. Render the unified HTML code safely using components.html
        components.html(bar_chart_html, height=330, scrolling=False)

    with col_HM:
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
                df['prediction_probability'] = pipeline.predict_proba(X_safe)[:, 1]

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

          
            # Heatmap
            fig_heat = px.imshow(
                heatmap_pivot,
                labels=dict(
                    color="Avg Risk Probability"
                ),
                color_continuous_scale='RdYlGn_r',
                text_auto=".2f"
            )

            # ------------------------------------------------------------------
            # Layout
            # ------------------------------------------------------------------
            fig_heat.update_layout(
                height=200,

                # CHANGE: Increased the left margin slightly from 50 to 65 to make sure region labels like "North-East" don't get chopped off on the edge
                margin=dict(l=65, r=30, t=10, b=30),

                # CHANGE: Set both axis titles to None to completely remove the overlapping text layers
                xaxis_title=None,
                yaxis_title=None,

                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor="white",

                font=dict(size=10)
            )

            fig_heat.update_xaxes(showticklabels=False, visible=False)
            fig_heat.update_yaxes(showticklabels=False, visible=False)
            
            # 2. Convert the figure directly into raw HTML components
            HM_html = pio.to_html(fig_heat, include_plotlyjs='cdn', config={"displayModeBar": False})
        
            # 3. Create a clean HTML layout string
            heatmap_html = f"""
                <div style="
                    margin-top: 42px; 
                    background-color: #ffffff; 
                    padding: 15px; 
                    height: 270px; 
                    border-radius: 20px; 
                    border: 1px solid #e6e9ef;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
                    font-family: 'Arial Black', Arial, sans-serif;
                    box-sizing: border-box; 
                    overflow: hidden; 
                    ">
                    <h3 style="
                        font-size: 18px; 
                        color: #0B3C49; 
                        margin: 10px 0 15px 0; 
                        text-align: center;
                        background-color: transparent;
                    ">Risk Heatmap</h3> <div style="margin-top: 5px;"> 
                        {HM_html}
                    </div>
                    
                </div>
            """

            # 4. Render the unified HTML code safely using components.html
            components.html(heatmap_html, height=330, scrolling=False)

  
    # =====================================================================================
    # LOAN RISK PREDICTION TOOL
    # =====================================================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background-color:#F8FAFC;
    padding:20px;
    border-radius:12px;
    border:1px solid #E5E7EB;
    text-align:center;
    ">

    <h3 style="
    margin-bottom:0px;
    font-size:28px;
    color:#0B3C49;
    font-weight:900;
    font-family:'Arial Black';
    ">
    🎯 Loan Risk Prediction Tool
    </h3>

    <p style="
    color:gray;
    margin-top:2px;
    font-family:'Arial Black';
    font-size:18px;
    font-color: #0B3C49;
    ">
    Enter borrower information to predict probability of loan default.
    </p>

    </div>
    """, unsafe_allow_html=True)


    with st.form("loan_feature_form"):

        left_col, right_col, centre_col = st.columns(3)

        with left_col:
            st.markdown("""
            <style>
                /* 1. Target only Widget Labels for Arial Black */
                label[data-testid="stWidgetLabel"] p {
                        font-family: "Arial Black", Gadget, sans-serif !important;
                        color: #0B3C49 !important;
                }

                /* 2. Force Buttons to ignore the styling and keep their original font */
                div.stButton > button {
                        font-family: "Source Sans Pro", sans-serif !important; /* Streamlit's default */
                }
            </style>
            """, 
            unsafe_allow_html=True)
        

            st.markdown(
                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif;
                font-size: 24px; 
                font-weight: 900;
                margin-bottom: 10px;
                ">
                1. Demographics
                </div>
                """, 
                unsafe_allow_html=True
            )

            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=70,
                value=30,
                step=1,
                format="%d",
                key="prediction_age"
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="prediction_gender"
            )

            region = st.selectbox(
                "Region",
                ["North", "South", "East", "West"],
                key="prediction_region"
            )

            st.markdown(
                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: 24px; 
                font-weight: 900;
                margin-bottom: 10px;
                ">
                2. Loan Information
                </div>
                """, 
                unsafe_allow_html=True
            )

            loan_type = st.selectbox(

                "Loan Type", 
                ["Type 1", "Type 2", "Type 3"],
                key="prediction_loan_type"
            )

            
            loan_amount = st.number_input(
                "Loan Amount (KES)",
                min_value=0,
                value=100000,
                key="prediction_loan_amount"
            )

            term = st.number_input(
                "Loan Term (Months)",
                min_value=6,
                max_value=360,
                value=24,
                step=1,
                key="prediction_term"
            )

            loan_limit = st.selectbox(
                "Loan Limit", 
                ["CF", "NCF"],
                key="prediction_loan_limit"
            )

            approv_in_adv = st.selectbox(
                "Approved in Advance",
                ["Yes", "No"],
                key="prediction_approv_in_adv"
            )

            submission_of_application = st.selectbox(
                "Submission Type",
                ["Online", "In-Person"],
                key="prediction_submission_type"
            )
            
            property_value = st.number_input(
                "Property Value (KES)",
                min_value=0,
                value=2000000,
                key="prediction_property_value"
            )

        with right_col:
            
            st.markdown(

                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: clamp(18px, 4vw, 24px);
                font-weight: 900;
                margin-top: 10px;
                margin-bottom: 10px;
                ">
                3. Credit Information
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            Credit_Worthiness = st.selectbox(
                "Credit Worthiness", 
                ["High", "Low"],
                key="prediction_credit_worthiness"
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                step=1,
                key="prediction_credit_score"
            )

            credit_type = st.selectbox(
                "Credit Type", 
                ["CI", "CRIF", "EQUI", "EXP"],
                key="prediction_credit_type"
            )

            co_applicant_credit_type = st.selectbox(
                "Co-applicant Credit Type",
                ["CIB", "EXP"],
                key="prediction_co_applicant_credit_type"
            )

            open_credit = st.selectbox(
                "Open Credit",
                ["opc", "nopc"],
                key="prediction_open_credit"
            )
            st.markdown(

                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: clamp(18px, 4vw, 24px);
                font-weight: 900;
                margin-top: 10px;
                ">
                4. Financial Capacity
                </div>
                """, 
                unsafe_allow_html=True
            )

            income = st.number_input(
                "Monthly Income (KES)",
                min_value=0,
                value=50000,
                key="prediction_income"
            )

            dti = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.01,
                format="%.2f",
                key="input_dti_ratio"
            )
        
            ltv = st.number_input(
                "Loan-to-Value Ratio (LTV)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.01,
                format="%.2f",
                key="input_ltv_ratio"
            )

            st.markdown(

                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: clamp(18px, 4vw, 24px);
                font-weight: 900;
                margin-top: 10px;
                ">
                5. Application Metadata
                </div>
                """, 
                unsafe_allow_html=True
            )

            year = st.number_input(
                "Year the loan was issued ", 
                min_value=1900, 
                max_value=2030, 
                value=2026,
                key="prediction_year"
            )
    
    
        with centre_col:
            st.markdown(
                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: clamp(18px, 4vw, 24px);
                font-weight: 900;
                margin-top: 10px;
                margin-bottom: 10px;
                ">
                5. Property Information
                </div>
                """, 
                unsafe_allow_html=True
            )

            occupancy_type = st.selectbox(
                "Occupancy Type",
                ["ir", "pr", "sr"],
                key="prediction_occupancy_type"
            )

            construction_type = st.selectbox(
                "Construction Type",
                ["mh", "sb"],
                key="prediction_construction_type"
            )

            total_units = st.selectbox(
                "Total Units",
                ["1U", "2U", "3U", "4U"],
                key="prediction_total_units"
            )

            Secured_by = st.selectbox(
                "Secured By",
                ["home", "land"],
                key="prediction_secured_by"
            )

            Security_Type = st.selectbox(
                "Security Type",
                ["direct", "Indriect"],
                key="prediction_security_type"
            )

            st.markdown(
                """
                <div style="
                font-family: 'Arial Black', Gadget, sans-serif; 
                font-size: clamp(18px, 4vw, 24px);
                font-weight: 900;
                margin-top: 10px;
                margin-bottom: 10px;
                ">
                6. Loan Features
                </div>
                """, 
                unsafe_allow_html=True
            )
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["p1", "p2", "p3", "p4"],
                key="prediction_loan_purpose"
                )

            interest_only = st.selectbox(
                "Interest Only Loan", 
                ["int_only", "not_int"],
                key="prediction_interest_only"
            )

            lump_sum_payment = st.selectbox(
                "Lump Sum Payment", 
                ["lpsm", "not_lpsm"],
                key="prediction_lump_sum_payment"
            )

            Neg_ammortization = st.selectbox(
                "Negative Amortization", 
                ["neg_amm", "not_neg"],
                key="prediction_neg_ammortization"
            )

            business_or_commercial = st.selectbox(
                "Business/Commercial Loan",
                ["b/c", "nob/c"],
                key="prediction_business_commercial"

            )
            st.markdown("""
            <style>
            /* Center the entire form submit button container */
            div[data-testid="stFormSubmitButton"] {
                display: flex;
                justify-content: center;
            }

            /* Style the actual button */
            div[data-testid="stFormSubmitButton"] button {
                width: 230px !important;
                height: 50px !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                color: white !important;
                background-color: #0B3C49 !important;
                border-radius: 12px !important;
                transition: all 0.3s ease !important;
            }

            /* Hover */
            div[data-testid="stFormSubmitButton"] button:hover {
                background-color: #E53935 !important;
                cursor: pointer !important;
            }

            /* Active */
            div[data-testid="stFormSubmitButton"] button:active {
                background-color: #E53935 !important;
            }
            </style>
            """, unsafe_allow_html=True)

        submit_button = st.form_submit_button("🔍 Predict Risk")
        

    if submit_button:
    # Creating the input DataFrame with all 29 features
        input_data = pd.DataFrame({
            'year': [year],
            'loan_limit': [loan_limit],
            'Gender': [gender],
            'approv_in_adv': [approv_in_adv],
            'loan_type': [loan_type],
            'loan_purpose': [loan_purpose],
            'Credit_Worthiness': [Credit_Worthiness],
            'open_credit': [open_credit],
            'business_or_commercial': [business_or_commercial],
            'loan_amount': [loan_amount],
            'term': [term],
            'Neg_ammortization': [Neg_ammortization],
            'interest_only': [interest_only],
            'lump_sum_payment': [lump_sum_payment],
            'property_value': [property_value],
            'construction_type': [construction_type],
            'occupancy_type': [occupancy_type],
            'Secured_by': [Secured_by],
            'total_units': [total_units],
            'income': [income],
            'credit_type': [credit_type],
            'Credit_Score': [credit_score],
            'co-applicant_credit_type': [co_applicant_credit_type],
            'age': [age],
            'submission_of_application': [submission_of_application],
            'LTV': [ltv],
            'Region': [region],
            'Security_Type': [Security_Type],
            'dtir1': [dti]
        })

            # clean column names
        input_data.columns = input_data.columns.str.strip()

    
            # clean column values
        for col in input_data.columns: 
            input_data[col] = input_data[col].astype(str).str.strip()

            # 2. Ensure column order matches X_train
        input_data = input_data[feature_cols] 
            
            # 3. Get the probability of default (Class 1) using your pipeline
        prob_default = pipeline.predict_proba(input_data)[0][1] 
        prob_percent = prob_default * 100
    
            # Dynamic threshold logic for colors and text
        if prob_default < 0.30:
            status_color, bg_color, risk_text = "#2ECC71", "#E8F8F0", "LOW RISK"
            sub_text = "Loan is Likely to be Repaid"
            icon = "✓"

        elif prob_default < 0.70:
            status_color, bg_color, risk_text = "#F39C12", "#FEF5E7", "MEDIUM RISK"
            sub_text = "Caution Recommended"
            icon = "!"

        else:
            status_color, bg_color, risk_text = "#E74C3C", "#FDEDEC", "HIGH RISK"
            sub_text = "High Default Probability"
            icon = "!"
    
            # Divider
        st.markdown("<br><hr>", unsafe_allow_html=True)
        
             
        st.markdown(f""" 
                <div style="
                    background-color: {bg_color}; 
                    color: {status_color}; 
                    padding: 10px 15px; 
                    border-radius: 8px; 
                    font-weight: bold; 
                    font-size: 30px;
                    text-align: center;
                    margin-bottom: 10px;
                    font-family: 'Arial Black', sans-serif;
                    ">
                    {icon} {risk_text}
                </div>
                    <div style="
                    text-align: center; 
                    color: #0B3C49; 
                    font-size: 16px; 
                    font-family: 'Arial Black', sans-serif;
                    ">
                    {sub_text}
                    </div>
        """, unsafe_allow_html=True)

        st.markdown("""
                    <div style="
                    font-family:'Arial Black', sans-serif; 
                    font-size:22px; 
                    color:#0B3C49; 
                    font-weight:900; 
                    text-align: center;
                    margin-bottom:15px;
                    text-align: center;
                    ">
                    📊 Prediction Result
                    </div>
                """, unsafe_allow_html=True)
            
        risk_value = prob_default * 100
        prob_percent = risk_value
                
        col_metric, col, col_gauge = st.columns([0.33, 0.33, 0.33])

        with col_metric:

            metric_html = f"""
            <div style="
            background-color: #FFFFFF;
            padding: 25px;
            margin-top: 10px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.03);
            height: 270px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
            ">
            <div style="
            color: #374151;
            font-weight: 700;
            font-family: 'Arial Black', sans-serif;
            font-size: 15px;
            margin-bottom: 8px;
            ">
            Probability of Default
            </div>
            <div style="
            width: 40px;
            height: 3px;
            background-color: {status_color};
            border-radius: 2px;
            margin-bottom: 15px;
            ">
            </div>

            <div style="
            color: {status_color};
            font-family: 'Arial Black';
            font-size: 46px;
            font-weight: 900;
            line-height: 1;
            ">
            {prob_percent:.1f}%
            </div>

            <div style="
            margin-top: 20px;
            color: #6B7280;
            font-size: 14px;
            font-weight: 700;
            font-family: 'Arial Black';
            ">
            Model predicted default probability
            </div>
            </div>
            """
            st.markdown(textwrap.dedent(metric_html), unsafe_allow_html=True)

        with col_gauge:
        
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_value,
                number={
                    "suffix": "%",
                    "font": {"family": "Arial Black, Arial, sans-serif", "size": 30, "color": "#0B3C49"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickmode": "array",
                        "tickvals": [0, 25, 50, 75, 100],
                        "ticktext": ["0", "25", "50", "75", "100"],
                        "dtick": 50, 
                        "tickcolor": "black",
                        "tickfont": {"family": "Arial-Bold, Arial, sans-serif", 
                                     "size": 14,
                                       "color": "#0B3C49"}
                     
                    },
                    "bar": {"color": "#0B3C49"},
                    "steps": [
                        {"range": [0, 30], "color": "#2ECC71"},
                        {"range": [30, 70], "color": "#F1C40F"},
                        {"range": [70, 100], "color": "#E74C3C"}
                    ]
                }
            ))

            fig.update_layout(
                height=240,  
                margin=dict(l=40, r=40, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            # 2. Convert the figure to raw HTML components
            gauge_html = pio.to_html(fig, include_plotlyjs='cdn', config={"displayModeBar": False})

            # 3. Create a clean HTML layout string
            card_html = f"""
                <div style="
                    background-color: #ffffff; 
                    padding: 10px; 
                    height: 270px;
                    border-radius: 20px; 
                    border: 1px solid #e6e9ef;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
                    font-family: 'Arial Black', Arial, sans-serif;
                    overflow: hidden;
                    ">
                    <h3 style="
                        font-size: 16px; 
                        color: #0B3C49; 
                        margin: 0 0 10px 0; 
                        text-align: center;
                        background-color: transparent;
                    ">Loan Risk Gauge</h3>
                    
                    {gauge_html}
                    
                </div>
            """

            # 4. Render the unified HTML code safely 
            components.html(card_html, height=300, scrolling=False)