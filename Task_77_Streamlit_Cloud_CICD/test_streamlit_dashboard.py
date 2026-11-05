from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    ROOT / 'Task_76_Accessible_Custom_CSS_Dashboard' / 'app.py',
    ROOT / 'Task_80_API_Latency_Profiling' / 'app.py',
    ROOT / 'Task_81_Final_Project_Implementation_Report' / 'app.py',
    ROOT / 'Task_82_Manual_Multiple_Imputation' / 'app.py',
]


def test_streamlit_cloud_entrypoints_run():
    for entrypoint in ENTRYPOINTS:
        result = AppTest.from_file(str(entrypoint), default_timeout=180).run()
        assert not result.exception, f'{entrypoint}: {result.exception}'