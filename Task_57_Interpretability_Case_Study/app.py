from pathlib import Path

import streamlit as st


st.title('Task 57: Model Interpretability Case Study')
report_path = Path(__file__).with_name('CASE_STUDY.md')
report_text = report_path.read_text(encoding='utf-8')
st.markdown(report_text)
st.info('No clinician or domain-expert interviews were conducted for this educational repository. The report distinguishes domain-informed review considerations from direct expert testimony and includes questions for a real validation panel.')
st.download_button('Download the full case study', report_text, 'interpretability_case_study.md', 'text/markdown')