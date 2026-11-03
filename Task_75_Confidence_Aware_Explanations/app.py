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


def explanation_text(probability, contributions, feature_names):
    predicted_class = int(probability >= 0.5)
    confidence = max(probability, 1 - probability)
    if confidence >= 0.8:
        confidence_text = 'The model is relatively confident in this classification.'
    elif confidence >= 0.65:
        confidence_text = 'The model has moderate confidence; review the contributing inputs.'
    else:
        confidence_text = 'The model is uncertain. Do not rely on this result without further assessment.'
    ranking = np.argsort(np.abs(contributions))[::-1]
    main_reasons = []
    for index in ranking[:3]:
        direction = 'increases' if contributions[index] > 0 else 'decreases'
        main_reasons.append(f'{html.escape(feature_names[index])} {direction} the model score relative to the training mean')
    sentence = '; '.join(main_reasons) if main_reasons else 'No substantial feature contribution was found.'
    class_text = 'positive' if predicted_class else 'negative'
    return f'The model predicts the {class_text} class with probability {probability:.1%}. {confidence_text} The largest score contributions are: {sentence}. These are model associations, not causes.'


st.title('Task 75: Confidence-Aware Prediction Explanations')
st.caption('Human-readable explanations are generated from the scratch linear model’s exact logit contributions.')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, seed=75)
means, scales = standardize_fit(X_train)
X_train_scaled, X_test_scaled = standardize(X_train, means, scales), standardize(X_test, means, scales)
model = LogisticRegressionScratch(learning_rate=0.06, iterations=1400, l2=0.02).fit(X_train_scaled, y_train)
case_index = st.selectbox('Example held-out case', list(range(min(20, len(X_test_scaled)))))
row = X_test_scaled[case_index]
probability = float(model.predict_proba(row[None, :])[0])
contributions = model.weights * row
text = explanation_text(probability, contributions, features)
st.markdown(f'<div class="explanation-card">{text}</div>', unsafe_allow_html=True)
contribution_table = pd.DataFrame({'Feature': features, 'Standardized value': row, 'Logit contribution': contributions}).sort_values('Logit contribution', key=np.abs, ascending=False)
st.subheader('Contribution detail')
st.dataframe(contribution_table.set_index('Feature').style.format('{:.3f}'))
st.caption(f'Intercept contribution: {model.bias:.3f}. Confidence is max(p, 1−p) and describes this model’s output, not the probability that the explanation is correct.')