You've asked for the full code, likely referring to the complete set of files needed for your Streamlit HR Dashboard, including the necessary fixes.

I will provide the corrected code for the three main files: **`App.py`**, **`utils.py`** (with the critical pathing fix), and **`pages/performance_analytics.py`**.

-----

## 1\. `requirements.txt`

This file ensures Streamlit Community Cloud installs all the necessary packages.

```text
streamlit
pandas
numpy
plotly
openpyxl  # Often required by pandas for certain file types, good to include
```

-----

## 2\. `App.py` (Main Application Script)

This file sets up the page configuration, loads the data, and handles navigation between your pages.

```python
import os, sys
# ensure local package folder is preferred (avoids name collision with installed `utils` package)
# This helps Python find your local 'utils.py'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from utils import load_data

st.set_page_config(page_title='HR Dashboard', layout='wide')
st.title('HR Dashboard')

st.markdown("\nWelcome — use the left sidebar to navigate pages.\n")

# Load data once and store in session_state
if 'df' not in st.session_state:
    # IMPORTANT: Pass the filename here. The logic to find the full path is now inside load_data.
    with st.spinner('Loading data...'):
        st.session_state.df = load_data('HRDataset_v14.csv')
        st.success('Data loaded')

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', [
    'Employee Overview',
    'Performance Analytics',
    'Salary Analytics'
])

# Import pages lazily based on selection
if page == 'Employee Overview':
    import pages.employee_overview as page_mod
elif page == 'Performance Analytics':
    import pages.performance_analytics as page_mod
elif page == 'Salary Analytics':
    import pages.salary_analytics as page_mod

# Run the selected page module
page_mod.run(st.session_state.df)
```

-----

## 3\. `utils.py` (Utilities with File Path Fix)

This file contains the data loading logic, including the **critical fix** to ensure `HRDataset_v14.csv` is found in the cloud environment using `os.path`.

```python
import os # 👈 NEW IMPORT
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

def load_data(filename: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Basic data loader for CSVs. Returns dataframe with common date columns parsed.
    - filename: Name of the csv file (e.g., 'HRDataset_v14.csv')
    - parse_dates: optional list of columns to parse as dates
    """
    # 💡 FIX: Use the absolute path relative to the location of this script (utils.py)
    # 1. Get the directory where utils.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construct the full path using the directory and the filename argument
    full_path = os.path.join(current_dir, filename) 
    
    # Use the calculated full_path for robust file access
    df = pd.read_csv(full_path) 
    
    # If user provided parse_dates, use them; otherwise try defaults
    to_parse = parse_dates if parse_dates is not None else [c for c in DEFAULT_DATE_COLS if c in df.columns]
    if to_parse:
        ensure_datetime(df, to_parse)
        
    return df
```

-----

## 4\. `pages/performance_analytics.py`

This file is a page module and requires no changes, but is included for completeness. **You will also need to ensure your `pages/employee_overview.py` and `pages/salary_analytics.py` files exist.**

```python
import re
from collections import Counter
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Mapping of common performance phrases to numeric scores (adjust to your org's scale)
_PERF_MAP = {
    'outstanding': 5,
    'exceeds expectations': 4,
    'exceeds': 4,
    'fully meets': 3,
    'meets': 3,
    'partially meets': 2,
    'partially': 2,
    'does not meet': 1,
    'doesnotmeet': 1,
    'below expectations': 1,
}


def _clean_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)  # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _extract_most_likely_label(s: str, known_labels: Optional[List[str]] = None) -> str:
    """
    If string looks concatenated (e.g. 'Fully MeetsFully Meets...'), try to find known labels inside it.
    Returns the matched known label or a best-effort token.
    """
    if not isinstance(s, str) or not s:
        return ''
    s_clean = _clean_text(s)
    if known_labels is None:
        known_labels = list(_PERF_MAP.keys())
    matches = [lbl for lbl in known_labels if lbl in s_clean]
    if matches:
        # return the most frequent match (though typically only one)
        return Counter(matches).most_common(1)[0][0]
    # fallback: return first two tokens joined (e.g., 'fully meets' -> 'fully meets')
    tokens = s_clean.split()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0] if tokens else ''


def normalize_score_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series with mixed numeric/text performance values into float scores.
    Strategy:
      1. Try to coerce to numeric.
      2. Map known phrases to numbers.
      3. Extract labels from messy strings and map.
      4. Last resort: categorical codes (1-based).
    Returns float Series (NaN where impossible).
    """
    # 1) numeric coercion
    num = pd.to_numeric(s, errors='coerce')
    # If a decent portion are numeric, treat series as numeric
    if num.notna().sum() >= max(1, int(0.2 * len(s))):
        return num.astype('float64')

    # 2) attempt phrase mapping after cleaning
    s_str = s.fillna('').astype(str)
    mapped = s_str.map(lambda x: _PERF_MAP.get(_clean_text(x), np.nan))
    if mapped.notna().sum() > 0:
        return mapped.astype('float64')

    # 3) extract likely label and remap
    extracted = s_str.map(lambda x: _extract_most_likely_label(x, known_labels=list(_PERF_MAP.keys())))
    mapped2 = extracted.map(lambda x: _PERF_MAP.get(x, np.nan))
    if mapped2.notna().sum() > 0:
        return mapped2.astype('float64')

    # 4) fallback to categorical codes (stable numeric representation)
    try:
        # Replace empty strings with NaN so codes for '' become -1 -> np.nan
        s_cleaned = s_str.replace('', np.nan)
        cat = pd.Categorical(s_cleaned)
        codes = pd.Series(cat.codes, index=s.index).replace(-1, np.nan).astype('float64')
        if codes.notna().any():
            # shift codes to start at 1 for nicer display (optional)
            codes = codes + 1.0
        return codes
    except Exception:
        return pd.Series(np.nan, index=s.index, dtype='float64')


def run(df: pd.DataFrame):
    st.header('Performance Analytics')

    if df is None or df.empty:
        st.info("No data available.")
        return

    # find a candidate score column
    score_col = None
    for cand in ['PerformanceScore', 'PerfScoreID', 'PerfScoreID ']:
        if cand in df.columns:
            score_col = cand
            break

    if score_col is None:
        st.info('No performance score column found.')
        return

    # Normalize score column into numeric values (new temporary column)
    df = df.copy()
    df['_normalized_score'] = normalize_score_series(df[score_col])

    # basic KPIs
    st.subheader('KPIs')
    
    # Use columns for cleaner display
    col1, col2, col3 = st.columns(3)

    if df['_normalized_score'].notna().sum() > 0:
        avg = df['_normalized_score'].mean()
        mx = df['_normalized_score'].max()
        mn = df['_normalized_score'].min()
        
        with col1:
            st.metric('Avg Score', round(avg, 2))
        with col2:
            st.metric('Max Score', float(mx) if pd.notna(mx) else '—')
        with col3:
            st.metric('Min Score', float(mn) if pd.notna(mn) else '—')
    else:
        st.info('No numeric performance scores could be derived from the selected score column.')
        st.write('Unique raw values (sample):')
        st.write(df[score_col].astype(str).dropna().unique()[:30])
        # still continue but many visualizations will be skipped

    # Performance by department
    dept_col = next((c for c in ['Department', 'DeptID', 'DepartmentName'] if c in df.columns), None)
    if dept_col:
        # If normalized scores exist, group by mean; otherwise show counts
        if df['_normalized_score'].notna().sum() > 0:
            perf_dept = (
                df.groupby(dept_col)['_normalized_score']
                    .mean()
                    .reset_index()
                    .rename(columns={'_normalized_score': score_col})
                    .sort_values(score_col, ascending=False)
            )
            st.subheader('Avg Performance by Department')
            fig = px.bar(perf_dept, x=dept_col, y=score_col, title='Avg Performance by Dept')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader('Performance counts by Department')
            perf_dept_counts = df.groupby(dept_col)[score_col].value_counts().unstack(fill_value=0)
            st.dataframe(perf_dept_counts)

    # Salary vs Performance scatter (only if numeric scores exist)
    if 'Salary' in df.columns and df['_normalized_score'].notna().sum() > 0:
        st.subheader('Salary vs Performance')
        tmp = df[[ '_normalized_score', 'Salary']].dropna()
        # try to coerce salary to numeric for plotting
        tmp['Salary'] = pd.to_numeric(tmp['Salary'], errors='coerce')
        tmp = tmp.dropna()
        if not tmp.empty:
            fig = px.scatter(tmp, x='Salary', y='_normalized_score', trendline='ols', title='Salary vs Performance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write('Salary data not numeric or missing for plotting.')

    # Top/Bottom performers
    st.subheader('Top & Bottom Performers')
    display_cols = []
    for c in ['Employee_Name', score_col, 'Department']:
        if c in df.columns or c == score_col:
            # keep score_col even if not present in original, to show normalized numeric
            if c in df.columns:
                display_cols.append(c)
            elif c == score_col:
                display_cols.append(score_col)

    # If numeric normalized scores available, use them for ranking; else show category counts
    if df['_normalized_score'].notna().sum() > 0:
        # prepare a display frame with normalized score
        disp = df.copy()
        disp['_display_score'] = disp['_normalized_score']
        # show top/bottom by normalized numeric score
        if not disp.dropna(subset=['_display_score']).empty:
            top = disp.nlargest(5, '_display_score')[display_cols + ['_normalized_score']]
            bot = disp.nsmallest(5, '_display_score')[display_cols + ['_normalized_score']]
            st.write('Top performers (by normalized score)')
            st.table(top.reset_index(drop=True))
            st.write('Bottom performers (by normalized score)')
            st.table(bot.reset_index(drop=True))
        else:
            st.write('No valid numeric scores to determine top/bottom performers.')
    else:
        st.write('Performance is non-numeric; showing counts by category.')
        st.write(df[score_col].value_counts())

    # cleanup temporary column
    if '_normalized_score' in df.columns:
        # nothing to do since df was a copy, but if you reused original, consider dropping
        pass
```
