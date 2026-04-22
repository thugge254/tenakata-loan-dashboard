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