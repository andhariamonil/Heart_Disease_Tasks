# Heart Disease Prediction Using Machine Learning

## Task 77: Streamlit Cloud CI/CD

### Objective

The repository workflow `.github/workflows/streamlit-ci.yml` compiles the selected Streamlit dashboard and inference API and runs protected API tests on pull requests and pushes to `main`. Streamlit Community Cloud handles automatic rebuilds after the repository and branch are connected; this workflow is a quality gate, not a Streamlit deployment API.

### Streamlit Cloud setup

1. Push this repository to GitHub.
2. Create a Streamlit Cloud app from the repository’s `main` branch.
3. Set the entry point to `Task_76_Accessible_Custom_CSS_Dashboard/app.py` and use the repository-root `requirements.txt`.
4. Configure secrets in Streamlit’s app settings.
5. Protect `main` and require the `Streamlit Cloud CI / validate` check.

This environment has no GitHub or Streamlit Cloud credentials, so repository connection and first publication must be completed in your account.

No notebook is included: the GitHub workflow and account-connection steps are operational configuration rather than notebook analysis.

### Run the guide app

```bash
source .venv/bin/activate
streamlit run Task_77_Streamlit_Cloud_CICD/app.py
```