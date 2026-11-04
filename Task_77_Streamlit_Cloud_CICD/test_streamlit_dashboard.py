from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / 'Task_76_Accessible_Custom_CSS_Dashboard' / 'app.py'


def test_streamlit_cloud_entrypoint_runs():
    result = AppTest.from_file(str(DASHBOARD), default_timeout=120).run()
    assert not result.exception