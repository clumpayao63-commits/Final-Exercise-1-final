import streamlit as st
import pandas as pd
from modules.data_processing import validate_dataframe
from modules.exploratory_analysis import show_eda
from modules.prediction import show_prediction
from modules.dashboard import show_dashboard

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="📋", layout="wide")
st.title("Student Service Request Monitoring System")
st.caption("SLA breach prediction and decision-support dashboard")

menu = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])

if "data" not in st.session_state:
    st.session_state.data = None

if menu == "Home":
    st.subheader("Welcome")
    st.write("Upload a student service request CSV, explore request patterns, and use the saved midterm model to support staff review.")
    st.info("This application supports staff decisions. Predictions must not automatically approve, reject, or deny student services.")
    st.write("Start with **Dataset** to upload your CSV.")
elif menu == "Dataset":
    st.subheader("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student service requests (CSV)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            report = validate_dataframe(df)
            st.session_state.data = df
            st.success("CSV loaded successfully.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(df))
            c2.metric("Columns", len(df.columns))
            c3.metric("Duplicate rows", int(df.duplicated().sum()))
            st.write("Data preview")
            st.dataframe(df.head(50), use_container_width=True)
            st.write("Missing values by column")
            st.dataframe(report, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    elif st.session_state.data is not None:
        st.dataframe(st.session_state.data.head(50), use_container_width=True)
    else:
        st.info("Upload a CSV file to begin.")
elif menu == "Data Analysis":
    if st.session_state.data is None:
        st.warning("Upload a CSV first in the Dataset page.")
    else:
        show_eda(st.session_state.data)
elif menu == "Prediction":
    show_prediction(st.session_state.data)
elif menu == "Dashboard":
    show_dashboard(st.session_state.data)
elif menu == "System Design":
    st.subheader("System Design (SAD Integration)")
    st.markdown("""
### Actors
- **Student Services Staff:** uploads datasets, reviews predictions and records final decisions.
- **System Administrator:** maintains the application and manages authorized access.

### Functional requirements
1. The system shall allow users to upload CSV datasets.
2. The system shall validate columns and report missing and duplicate values.
3. The system shall display descriptive statistics and interactive charts.
4. The system shall predict possible SLA breaches using the saved trained pipeline.
5. The system shall display monitoring KPIs, risk cases and recommended staff follow-up.

### Non-functional requirements
1. The interface should be easy to use.
2. Data processing and predictions should complete efficiently for supported files.
3. Student information must be protected and access restricted to authorized users.

### Decision-support workflow
Upload CSV → Validate data → Explore patterns → Generate prediction → Review dashboard → Staff records the final decision.

### Human review
A prediction is only a decision-support signal. Authorized staff must review the case and make the final decision.
""")
    st.info("Create and save your updated use-case diagram as assets/use_case_diagram.png, then add it to this page or your report.")
