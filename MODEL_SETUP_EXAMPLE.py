# Adapt this snippet to the variable names in your ORIGINAL midterm notebook.
# Run after fitting/evaluating the original preprocessing + model pipeline.
import joblib

bundle = {
    "pipeline": fitted_pipeline,  # your already-fitted preprocessing + estimator pipeline
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
