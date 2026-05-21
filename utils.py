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

TEXT_COLOR = "#0B3C49"
NUMBER_COLOR = "#E53935"
ACCENT_COLOR = "#FFB300"

def Apply_purpose_map(df, purpose_map):
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