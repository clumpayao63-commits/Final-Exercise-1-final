import pandas as pd
import streamlit as st
from modules import data_processing as dp, exploratory_analysis as eda, prediction, dashboard

st.set_page_config(page_title="SLA Breach Monitoring - Student Services", layout="wide")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Synthetic data only. Predictions support, not replace, staff decisions.")

def current_df():
    """Uploaded data if valid, otherwise the bundled synthetic sample."""
    return st.session_state.get("df", dp.load_sample())

if menu == "Home":
    meta = dp.load_metadata()
    st.title("Student Service Request Monitoring")
    st.write("Predicts which student service requests are at risk of breaching their SLA so staff can intervene early.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model version", meta["model_version"])
    c2.metric("ROC AUC", f"{meta['metrics']['roc_auc']:.3f}")
    c3.metric("Recall (breach)", f"{meta['metrics']['recall']:.3f}")
    st.markdown("**Workflow:** Upload data → Explore → Predict → Review dashboard → Staff records decision.")
    st.warning("Decision support only: predictions never authorize denying or delaying student services.")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df, err = dp.read_uploaded_csv(up)
        if err:
            st.error(err)
        else:
            ok, problems = dp.validate_schema(df)
            if not ok:
                for p in problems:
                    st.error(p)
                st.info(f"Required columns: {', '.join(dp.REQUIRED_COLUMNS)}")
            else:
                st.success("Schema valid. Data loaded for all modules.")
                st.session_state["df"] = df
    else:
        st.info("No file uploaded - using the bundled synthetic sample.")
    df = current_df()
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Rows:", len(df), "| Columns:", df.shape[1])
    nulls, dups = dp.quality_summary(df)
    st.write("Duplicate rows:", dups)
    st.dataframe(nulls)

elif menu == "Data Analysis":
    eda.render(current_df())

elif menu == "Prediction":
    prediction.render()

elif menu == "Dashboard":
    dashboard.render(current_df())

else:
    st.header("System Analysis and Design")
    st.subheader("Functional requirements")
    st.markdown("""
1. **FR1** - Staff can upload a CSV of service requests and see schema, null and duplicate checks.
2. **FR2** - The system shows descriptive statistics, class distribution and interactive charts.
3. **FR3** - Staff can enter a request's details and receive a breach prediction with probability.
4. **FR4** - The dashboard ranks requests by breach risk and shows workload KPIs.
5. **FR5** - Authorized staff can record a final decision (with justification for overrides).
""")
    st.subheader("Non-functional requirements")
    st.markdown("""
- **NFR1 Performance** - Model is loaded once (cached); predictions return in under 2 seconds.
- **NFR2 Privacy/Security** - Only synthetic or authorized data; no credentials or student records in the repository.
- **NFR3 Transparency/Usability** - Model version, timestamp, limitations and human-review notice are always visible.
""")
    st.subheader("Actors and permissions")
    st.table(pd.DataFrame({
        "Actor": ["Student Services Staff", "Supervisor", "System Administrator", "Student (indirect)"],
        "Can view predictions": ["Yes", "Yes", "Yes (audit)", "No"],
        "Can record final decision": ["Yes", "Yes (overrides, escalations)", "No", "No"]}))
    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png")
    st.subheader("Analytical process / decision workflow")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style=rounded];
 A [label="Upload CSV"] -> B [label="Validate schema"] -> C [label="Preprocess + saved pipeline"] ->
 D [label="Breach probability"] -> E [label="Risk band / review queue"] -> F [label="Staff review"] -> G [label="Final decision recorded"];
 F -> H [label="Override + justification"]; }
""")
