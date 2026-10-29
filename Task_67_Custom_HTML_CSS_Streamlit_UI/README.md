# Heart Disease Prediction Using Machine Learning

## Task 67: Custom HTML & CSS Responsive Streamlit UI

### Objective

Build a responsive, modern Streamlit user interface **without utilizing any built-in Streamlit layout functions** (`st.columns`, `st.sidebar`, `st.expander`, `st.tabs`). Instead, all visual components, layouts, cards, and data representations are manually structured using semantic **HTML5** and custom **CSS3** via `st.markdown(..., unsafe_allow_html=True)`.

Key Architectural Highlights:
1. **Zero Built-In Layouts**: Complete elimination of `st.columns`, `st.sidebar`, `st.expander`, `st.tabs`, or container borders.
2. **CSS Grid & Flexbox Manual Structuring**:
   - Modern `.kpi-grid` and `.card-grid` utilizing `repeat(auto-fit, minmax(...))` for auto-wrapping.
   - Flexible navigation bar with pulse heartbeat animation and status pills.
   - Fluid typography using CSS `clamp()`.
3. **Cross-Device Responsiveness**:
   - **Desktop (>1024px)**: Multi-column 4-card KPI metrics, side-by-side diagnostic cards, balanced spacing.
   - **Tablet (768px - 1024px)**: 2-column wrapping grid with adjusted padding.
   - **Mobile (≤768px and ≤480px)**: Reflows seamlessly to a clean, vertical single-column stacked layout with touch-friendly form controls and horizontally scrollable tables.
4. **End-to-End Clinical Inference**:
   - Powered by the scratch-trained `LogisticRegressionScratch` classifier trained on standardized Cleveland Heart Disease patient records ($N = 302$).
   - Dynamic real-time risk assessment: Low Risk (emerald green), Moderate Risk (amber orange), High Risk (crimson red).
   - Dynamic CSS progress bar gauge and patient clinical risk factor attribution tags.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_67_Custom_HTML_CSS_Streamlit_UI/app.py
```

Run all cells in `Task_67_Custom_HTML_CSS_Streamlit_UI/Task_67_Custom_HTML_CSS_Streamlit_UI.ipynb` for the standalone notebook.

