# Student Service Request Monitoring System

A Streamlit decision-support application for exploring student service requests and using a saved midterm machine-learning pipeline to support SLA-breach review.

## Run locally
1. Install Python 3.10 or a compatible version.
2. Open this folder in VS Code.
3. Create and activate a virtual environment.
4. Install packages: `pip install -r requirements.txt`
5. Place your original fitted midterm pipeline at `models/trained_model.joblib`.
6. Run: `streamlit run app.py`

## Important: connect your actual midterm model
The assignment requires reuse of the original midterm dataset, target, features, preprocessing and trained model. Do not use a newly trained unrelated model.

Recommended saved bundle:
```python
import joblib
bundle = {
    "pipeline": fitted_pipeline,
    "feature_columns": X_train.columns.tolist(),
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    },
    "target_column": TARGET
}
joblib.dump(bundle, "models/trained_model.joblib")
```
Replace variable names with the exact names in your notebook. Ensure `fitted_pipeline` is already fitted and contains the preprocessing steps used during training.

## Required project evidence
- Original midterm notebook and dataset
- Screenshots of Home, Dataset, EDA, Prediction, Dashboard and System Design
- Five sample predictions
- Updated use-case diagram and decision workflow
- GitHub repository URL and deployed Streamlit URL
- Deployment report describing reused code, changes, tests, limitations and reflection

## Privacy and responsible use
Use synthetic or properly authorized data for public deployment. Do not commit passwords or sensitive student records. Predictions are decision-support outputs; authorized staff make final decisions.
