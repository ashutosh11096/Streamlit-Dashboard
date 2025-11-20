# pages/employee_overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import tenure_days

def run(df: pd.DataFrame):
    st.header('Employee Overview')

    # Basic KPIs
    total = df.shape[0]
    terminated = df['DateofTermination'].notna().sum() if 'DateofTermination' in df.columns else 0
    active = total - terminated

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Employees', total)
    col2.metric('Active Employees', active)
    col3.metric('Terminated Employees', terminated)

   
    st.write('---')

    # Demographics
    demo_cols = []
    for c in ['Gender', 'Sex', 'RaceDesc', 'MaritalDesc', 'CitizenDesc']:
        if c in df.columns:
            demo_cols.append(c)

    if demo_cols:
        st.subheader('Demographics')
        cols = st.columns(len(demo_cols))
        for i, c in enumerate(demo_cols):
            with cols[i]:
                counts = df[c].value_counts().reset_index()
                counts.columns = [c, 'count']
                fig = px.pie(counts, names=c, values='count', title=c)
                st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.subheader('Department headcount')
    dept_col = None
    for cand in ['Department', 'DeptID', 'DepartmentName']:
        if cand in df.columns:
            dept_col = cand
            break
    if dept_col:
        hc = df[dept_col].value_counts().reset_index()
        hc.columns = [dept_col, 'count']
        fig = px.bar(hc, x=dept_col, y='count', title='Headcount by Department')
        st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.subheader('Quick table')
    st.dataframe(df.head(50))
