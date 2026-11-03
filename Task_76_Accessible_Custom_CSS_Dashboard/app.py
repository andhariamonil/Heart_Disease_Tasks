import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, standardize, standardize_fit, stratified_split


st.set_page_config(page_title='Heart Model Review', page_icon='♥', layout='wide')
st.markdown('''
<style>
:root { color-scheme: light; }
.stApp { background: #f3f6f7; color: #172a35; }
[data-testid="stHeader"] { background: rgba(243,246,247,.94); }
section.main > div { max-width: 1080px; padding-top: 1.5rem; }
h1, h2, h3 { color: #123b4a; letter-spacing: 0; }
.review-hero { background: linear-gradient(120deg,#123b4a,#176b68); color: #fff; padding: 1.5rem 1.75rem; border-radius: 10px; margin: .5rem 0 1.25rem; animation: enter .45s ease-out both; }
.review-hero h1 { color: #fff; margin: 0 0 .4rem; font-size: 1.8rem; }
.review-hero p { color: #e4f2ef; margin: 0; }
.review-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .8rem; margin: 1rem 0; }
.review-card { background: #fff; border: 1px solid #d5e0e3; border-top: 4px solid #1c7972; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 8px rgba(18,59,74,.08); transition: transform .18s ease, box-shadow .18s ease; }
.review-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(18,59,74,.14); }
.review-label { color: #425b64; font-size: .85rem; }
.review-value { color: #123b4a; font-size: 1.65rem; font-weight: 700; margin-top: .25rem; }
.review-explanation { background: #e9f4f2; border-left: 4px solid #176b68; border-radius: 4px; padding: 1rem 1.1rem; color: #163d45; line-height: 1.55; margin: 1rem 0; }
.review-note { color: #425b64; font-size: .9rem; }
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button { background: #135c63; color: white; border: 2px solid #135c63; border-radius: 6px; min-height: 44px; transition: background .15s ease, transform .15s ease; }
div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover { background: #0c4349; border-color: #0c4349; transform: translateY(-1px); }
div[data-testid="stButton"] button:focus-visible, div[data-testid="stFormSubmitButton"] button:focus-visible, input:focus-visible { outline: 3px solid #d68137 !important; outline-offset: 2px; }
@keyframes enter { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@media (max-width: 700px) { .review-grid { grid-template-columns: 1fr; } .review-hero { padding: 1.1rem; } .review-hero h1 { font-size: 1.45rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; scroll-behavior: auto !important; } }
</style>
''', unsafe_allow_html=True)

csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=76)
means, scales = standardize_fit(X_train)
X_train_scaled, X_test_scaled = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)

st.markdown('<div class="review-hero"><h1>Heart Model Review</h1><p>Interactive prediction review with input checks, confidence context, and model-associated feature contributions.</p></div>', unsafe_allow_html=True)
selected = st.selectbox('Choose a held-out example', list(range(min(30, len(X_test_scaled)))))
row = X_test_scaled[selected]
probability = float(model.predict_proba(row[None, :])[0])
prediction = int(probability >= 0.5)
confidence = max(probability, 1 - probability)
contributions = model.weights * row
top_index = np.argsort(np.abs(contributions))[::-1][:3]
direction = 'positive' if prediction else 'negative'
if confidence >= .8:
    confidence_phrase = 'higher model confidence'
elif confidence >= .65:
    confidence_phrase = 'moderate model confidence'
else:
    confidence_phrase = 'low model confidence; human review is especially important'
reasons = ', '.join(f'{html.escape(features[i])} ({contributions[i]:+.2f} logit)' for i in top_index)
explanation = f'The model predicts the {direction} class with probability {probability:.1%} and {confidence_phrase}. Largest score associations: {reasons}. Contributions describe this fitted model, not causes.'
st.markdown(f'<div class="review-explanation">{explanation}</div>', unsafe_allow_html=True)
st.markdown(f'''<div class="review-grid">
<div class="review-card"><div class="review-label">Predicted class</div><div class="review-value">{direction.title()}</div></div>
<div class="review-card"><div class="review-label">Positive-class score</div><div class="review-value">{probability:.1%}</div></div>
<div class="review-card"><div class="review-label">Model confidence</div><div class="review-value">{confidence:.1%}</div></div>
</div>''', unsafe_allow_html=True)

with st.expander('Inspect feature contributions'):
    contributions_table = pd.DataFrame({'Feature': features, 'Standardized value': row, 'Logit contribution': contributions}).sort_values('Logit contribution', key=np.abs, ascending=False)
    st.dataframe(contributions_table.set_index('Feature').style.format('{:.3f}'))
st.markdown('<p class="review-note">Educational demonstration only. Not a diagnosis or substitute for professional review. Confidence describes model output, not certainty of a clinical outcome.</p>', unsafe_allow_html=True)