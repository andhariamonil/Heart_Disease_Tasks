# Heart Disease Prediction Using Machine Learning

## Task 71: UI Usability Testing, Heuristic Evaluation, and Iterative Refinement

### Objective

Conduct systematic user experience (UX) research on the heart disease prediction interface, gather qualitative and quantitative feedback across clinical user personas, identify interaction pain points using Jakob Nielsen's 10 Usability Heuristics, and iteratively refine the decision support interface to maximize clinical efficiency, error prevention, and user confidence.

Key Elements:
1. **Heuristic Evaluation Audit**:
   - Comprehensive audit across Jakob Nielsen's 10 Usability Heuristics (Visibility of system status, match with real world, error prevention, user freedom, recognition over recall, etc.).
   - Overall compliance score improved from **28/50 (56%)** to **47/50 (94%)**.
2. **Quantitative Usability Metrics**:
   - **System Usability Scale (SUS)** score increased from **58.2** (Grade D / marginal) to **86.8** (Grade A / excellent).
   - **Mean Task Completion Time** dropped by **66.5%** (from 48.4s to 16.2s).
   - **User Input Error Rate** declined from **24.3%** to **2.8%**.
3. **Iterative Refinements Implemented**:
   - Replaced cryptic integer codes (`cp`, `slope`, `ca`, `thal`) with standard American Heart Association medical nomenclature.
   - Integrated dynamic physiological alert badges (`⚠️ Stage 2 Hypertension`, `⚠️ High ST Depression`).
   - Grouped inputs into 3 logical clinical panels: Demographics, Stress ECG, and Fluoroscopy.
   - Introduced 1-click clinical archetype presets and instant form reset controls.
   - Implemented an interactive in-app usability telemetry collector and Before/After comparative review toggle.

Detailed findings are documented in [USABILITY_RESEARCH_REPORT.md](USABILITY_RESEARCH_REPORT.md).

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_71_UI_Usability_Testing_and_Refinement/app.py
```

Run all cells in `Task_71_UI_Usability_Testing_and_Refinement/Task_71_UI_Usability_Testing_and_Refinement.ipynb` for the standalone notebook.

