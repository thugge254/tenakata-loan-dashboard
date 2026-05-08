purpose_map = {
    "p1": "Consumption",
    "p2": "Education",
    "p3": "Business",
    "p4": "Property"
}

status_map = {
    0: "No Default",
    1: "Default"
}

def apply_purpose_mapping(df, purpose_map):
    df = df.copy()
    df["loan_purpose"] = df["loan_purpose"].map(purpose_map)
    return df

def calculate_total_loan_portfolio(df):
    return df["loan_amount"].sum()

def calculate_average_loan_size(df):
    return df["loan_amount"].mean()

def calculate_total_loans_issued(df):
    return len(df)

def calculate_average_interest_rate(df):
    return df["rate_of_interest"].mean()