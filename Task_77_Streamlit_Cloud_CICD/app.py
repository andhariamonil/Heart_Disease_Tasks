import streamlit as st


st.set_page_config(page_title='Deployment Guide', page_icon='⇧', layout='centered')
st.title('Task 77: Streamlit Cloud CI/CD')
st.write('GitHub Actions validates the selected Streamlit dashboard and API. Streamlit Cloud rebuilds the app when the connected repository branch updates.')
st.subheader('Deployment checklist')
st.markdown('''
1. Push this repository to GitHub with `main` as the deployment branch.
2. In Streamlit Community Cloud, create an app from this repository and select `Task_76_Accessible_Custom_CSS_Dashboard/app.py`.
3. Select `Task_76_Accessible_Custom_CSS_Dashboard/requirements.txt` for dependencies.
4. Configure any required secrets in Streamlit app settings, not in source files.
5. Protect `main` and require the `Streamlit Cloud CI / validate` check.
6. Merge an approved change; Streamlit Cloud rebuilds from the connected branch.
''')
st.info('The workflow validates code; Streamlit Cloud performs deployment. This workspace has no repository or deployment credentials, so it cannot connect or publish the app on your behalf.')
st.caption('Check repository visibility and access controls before publishing a private project.')