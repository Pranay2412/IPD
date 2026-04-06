import re
import pandas as pd

def extract_and_convert_dates_to_numeric(text: str) -> pd.Series:
    # Regex for timestamps like 'YYYY-MM-DD HH:MM:SS' or just date/time
    date_patterns = [
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # full datetime
        r"\d{4}-\d{2}-\d{2}",                     # date only
        r"\d{2}:\d{2}:\d{2}"                      # time only (less useful without date)
    ]
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates_found.extend(matches)
    
    dates_dt = pd.to_datetime(dates_found, errors='coerce')
    
    numeric_timestamps = dates_dt.astype('int64') / 1e9  # convert nanoseconds to seconds
    
    return numeric_timestamps
