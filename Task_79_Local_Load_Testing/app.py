import streamlit as st

from benchmark import MAX_REQUESTS, MAX_WORKERS, run_benchmark


st.title('Task 79: Bounded Local API Load Test')
st.warning('This tester is hard-bound to 127.0.0.1 and caps concurrency at 32 workers. Run only against your local API or an explicitly authorized staging environment.')
request_count = st.slider('Total requests', min_value=10, max_value=MAX_REQUESTS, value=200, step=10)
worker_count = st.slider('Concurrent local workers', min_value=1, max_value=MAX_WORKERS, value=16)
token = st.text_input('Local API bearer token', type='password')

if st.button('Run localhost load test', disabled=len(token) < 32):
    try:
        result = run_benchmark(request_count, worker_count, token)
        st.json(result)
    except Exception as error:
        st.error(f'Local load test could not complete: {error}')
st.caption('This bounded local batch measures throughput and latency, not thousands of simultaneous connections. Large-scale testing belongs on an authorized staging environment with provider quotas and a stop plan.')