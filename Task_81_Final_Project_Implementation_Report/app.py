from pathlib import Path

import streamlit as st


st.set_page_config(page_title='Project Implementation Report', layout='centered')
st.title('Task 81: Final Project Implementation Report')
report_path = Path(__file__).with_name('FINAL_PROJECT_REPORT.md')
report = report_path.read_text(encoding='utf-8')
st.markdown(report)
st.download_button('Download final project report', report, 'heart_disease_project_report.md', 'text/markdown')