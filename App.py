import os, sys
# ensure local package folder is preferred (avoids name collision with installed `utils` package)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from utils import load_data

st.set_page_config(page_title='HR Dashboard', layout='wide')
st.title('HR Dashboard')

st.markdown("\nWelcome — use the left sidebar to navigate pages.\n")

# Load data once and store in session_state
if 'df' not in st.session_state:
    # IMPORTANT: Pass the filename here. The logic to find the full path is now inside load_data in utils.py.
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
