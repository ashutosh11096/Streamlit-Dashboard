
# utils.py
import pandas as pd
import numpy as np
from typing import Optional, List

# Common date column names you might have in your HR dataset.
DEFAULT_DATE_COLS = [
    'DOB', 'DateofJoining', 'DateOfJoining', 'DOJ',
    'DateofTermination', 'DateOfTermination', 'TerminationDate'
]

def ensure_datetime(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert specified columns in df to datetime where possible.
    If cols is None, tries DEFAULT_DATE_COLS that exist in df.
    Returns the same dataframe (modified in-place).
    """
    if cols is None:
        cols = [c for c in DEFAULT_DATE_COLS if c in df.columns]
    # only convert columns that actually exist in df
    cols = [c for c in cols if c in df.columns]
    for c in cols:
        # coerce errors to NaT
        df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def tenure_days(df: pd.DataFrame,
                join_col_candidates: Optional[List[str]] = None,
                term_col_candidates: Optional[List[str]] = None) -> pd.Series:
    """
    Vectorized computation of tenure in days.
    Returns a pd.Series of dtype float64 with np.nan for missing values.
    - df: DataFrame
    - join_col_candidates: list of possible join date column names (defaults used if None)
    - term_col_candidates: list of possible termination date column names (defaults used if None)
    """
    if join_col_candidates is None:
        join_col_candidates = ['DateofJoining', 'DateOfJoining', 'DOJ', 'JoinDate', 'JoiningDate']
    if term_col_candidates is None:
        term_col_candidates = ['DateofTermination', 'DateOfTermination', 'TerminationDate', 'LeaveDate']

    # pick first candidate that exists in df
    join_col = next((c for c in join_col_candidates if c in df.columns), None)
    term_col = next((c for c in term_col_candidates if c in df.columns), None)

    if join_col is None:
        # Return a float Series of NaNs (length matches df)
        return pd.Series(np.nan, index=df.index, dtype='float64')

    # Ensure datetime for the relevant columns
    cols_to_convert = [join_col]
    if term_col:
        cols_to_convert.append(term_col)
    ensure_datetime(df, cols_to_convert)

    today = pd.Timestamp.today().normalize()

    jd = pd.to_datetime(df[join_col], errors='coerce')  # Timestamps or NaT
    if term_col and term_col in df.columns:
        ld = pd.to_datetime(df[term_col], errors='coerce')
    else:
        # create a Series filled with NaT so we can fill with today
        ld = pd.Series(pd.NaT, index=df.index)

    # For rows with no termination date, use today
    ld_filled = ld.fillna(today)

    # Compute difference in days (will be float after astype)
    diff = (ld_filled - jd).dt.days.astype('float64')  # NaN where jd is NaT

    # Set negative diffs (bad data) to NaN (change to 0.0 if you prefer)
    diff[diff < 0] = np.nan

    return diff

def load_data(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Basic data loader for CSVs. Returns dataframe with common date columns parsed.
    - path: path to csv or a file-like object (Streamlit uploader returns this)
    - parse_dates: optional list of columns to parse as dates
    """
    # pandas.read_csv accepts both a path (str) and file-like object
    df = pd.read_csv(path)
    # If user provided parse_dates, use them; otherwise try defaults
    to_parse = parse_dates if parse_dates is not None else [c for c in DEFAULT_DATE_COLS if c in df.columns]
    if to_parse:
        ensure_datetime(df, to_parse)
    return df
