import pandas as pd
import re
import numpy as np

def clean_market_data(raw_text):
    """
    Parses mixed financial text data into a Pandas DataFrame.
    """
    if not raw_text:
        return pd.DataFrame()

    # Remove unnecessary headers
    text = raw_text.replace("ParityT+F", "")
    
    # Protect index names ending with numbers (e.g., US30 -> US30|)
    known_indices = ["US30", "NAS100", "E DJI", "Em NQ", "NIKKEI JPN IND", "S&P500", "DAX", "^GDAXI"]
    for index in known_indices:
        text = text.replace(index, f"{index}|")

    # Regex Extraction
    # Logic: Name part | Value part
    pattern = r"([A-Za-z0-9/\s\(\)\.-]+?)(?:\|)?(-?\d+\.\d+|-?\d+|(?<=[A-Za-z])-|$)((?=[A-Z])|$)"
    matches = re.findall(pattern, text)

    # Clean regex output
    clean_matches = [[m[0].strip(), m[1]] for m in matches]

    # Create DataFrame (English Columns)
    df = pd.DataFrame(clean_matches, columns=["Instrument", "Value"])

    # Numeric Cleaning
    # Replace '-' or empty strings with NaN, convert rest to numbers
    df["Value"] = df["Value"].replace(["-", ""], np.nan)
    df["Value"] = pd.to_numeric(df["Value"], errors='coerce')

    return df
