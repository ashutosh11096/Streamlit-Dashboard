# pages/performance_analytics.py
import re
from collections import Counter
from typing import List, Optional

import importlib.util
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


def _statsmodels_available() -> bool:
    """Return True if statsmodels is importable in this environment."""
    return importlib.util.find_spec("statsmodels") is not None


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
    if df['_normalized_score'].notna().sum() > 0:
        avg = df['_normalized_score'].mean()
        mx = df['_normalized_score'].max()
        mn = df['_normalized_score'].min()
        st.metric('Avg Score', round(avg, 2))
        st.metric('Max Score', float(mx) if pd.notna(mx) else '—')
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
            st.plotly_chart(fig, width="stretch", key="perf_dept_bar")
        else:
            st.subheader('Performance counts by Department')
            perf_dept_counts = df.groupby(dept_col)[score_col].value_counts().unstack(fill_value=0)
            st.dataframe(perf_dept_counts)


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
