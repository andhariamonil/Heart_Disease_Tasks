# Heart Disease Prediction Using Machine Learning

## Task 4: Duplicate Row Detection and Removal

### Objective

Check for duplicate rows using a custom comparison algorithm, remove duplicates manually, and analyze their impact on data integrity and machine learning models.

### Requirements Implemented

- Manual CSV loading
- Custom row comparison algorithm
- Duplicate row detection
- Manual duplicate removal
- Data integrity analysis
- Machine learning impact analysis

### Implementation

Two rows are compared value by value using a custom comparison function.

Duplicate rows are removed manually by checking each row against the rows already stored in the cleaned dataset.

No Pandas duplicate detection or removal functions are used.

### Run

```bash
streamlit run app.py