# Student Service Request Monitoring - SLA Breach Prediction (Streamlit)

Deployment of the Midterm data-mining output (Section B) as a Streamlit app.

## Structure
app.py · requirements.txt · modules/ (data_processing, exploratory_analysis, prediction, dashboard) ·
models/trained_model.joblib + model_metadata.json · assets/use_case_diagram.png · data/ (synthetic sample) · notebooks/ (midterm .ipynb)

## Run locally
```
pip install -r requirements.txt
python train_model.py      # or export your midterm pipeline with joblib
python make_diagram.py
streamlit run app.py
```

## Midterm continuity
- Target: `sla_breached` (1 = breached). Prediction checkpoint: at request submission/assignment.
- Leakage removed: resolution_hours, closed_at, first_response_hours, final_status.
- Differences from the midterm: (WRITE YOURS HERE)

## Deploy
Push to GitHub → https://share.streamlit.io → New app → pick repo/branch → entry point `app.py`.

## Ethics
Synthetic data only. Predictions are decision support; staff make and record every final decision.
