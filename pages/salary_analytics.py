# pages/salary_analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px

def run(df: pd.DataFrame):
    st.header('Salary Analytics')
    if 'Salary' not in df.columns:
        st.info('No Salary column found')
        return

    # ensure we safely reference columns that may or may not exist
    cols_to_keep = [c for c in ['Employee_Name','Salary','Department','Gender','RaceDesc'] if c in df.columns]
    df_salary = df[cols_to_keep].copy()
    df_salary['Salary'] = pd.to_numeric(df_salary['Salary'], errors='coerce')

    st.subheader('KPIs')
    st.metric('Average Salary', f"{int(df_salary['Salary'].mean()):,}")
    st.metric('Median Salary', f"{int(df_salary['Salary'].median()):,}")
    st.metric('Max Salary', f"{int(df_salary['Salary'].max()):,}")

    st.write('---')
    st.subheader('Salary distribution')
    fig = px.histogram(df_salary, x='Salary', nbins=40, title='Salary Distribution')
    st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.subheader('Average salary by department')
    dept_col = next((c for c in ['Department','DeptID','DepartmentName'] if c in df.columns), None)
    if dept_col and 'Salary' in df_salary.columns:
        avg_dept = df_salary.groupby(dept_col)['Salary'].mean().reset_index().sort_values('Salary', ascending=False)
        fig2 = px.bar(avg_dept, x=dept_col, y='Salary', title='Avg Salary by Dept')
        st.plotly_chart(fig2, use_container_width=True)


