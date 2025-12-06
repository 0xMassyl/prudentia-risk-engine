import streamlit as st
import pandas as pd
import requests
import json
from src.domain.entities import ExposureType, Loan

# --- FastAPI Backend Configuration ---
# Make sure the API is running on this port (via uvicorn)
API_URL = "http://127.0.0.1:8000"

# --- UI Scenario Configuration ---
SCENARIOS_INFO = {
    "baseline": "Baseline (Stable growth, no shock applied)",
    "adverse": "Adverse (Mild slowdown, shock Z=1.5)",
    "severely_adverse": "Severely Adverse (Major crisis, shock Z=3.0)",
}

# --- Utility Functions ---
def call_api(endpoint: str, data: dict, scenario: str):
    """Performs a POST request to the FastAPI endpoint safely."""
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}?scenario={scenario}",
            data=json.dumps(data),
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text
            st.error(f"API Error (Code {response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Connection error: Make sure the FastAPI backend is running at {API_URL} (Terminal 1).")
        return None

# --- UI Construction ---
st.set_page_config(
    page_title="Prudentia Risk Engine UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏛️ Prudentia : Basel III/IV Risk Engine")
st.markdown("---")

# --- Input Form (Simplified Loan Portfolio) ---
st.sidebar.header("Demo Loan Configuration (Input)")

# User inputs
loan_id = st.sidebar.text_input("Loan ID", value="TEST_LOAN_001")

# IMPORTANT: variable named 'default_prob' to avoid shadowing 'pd' from pandas
default_prob = st.sidebar.slider("1. PD (Probability of Default)", 0.001, 0.15, 0.02, format="%.3f")
lgd = st.sidebar.slider("2. LGD (Loss Given Default)", 0.10, 0.80, 0.45, format="%.2f")
ead = st.sidebar.number_input("3. EAD (Exposure at Default) (€)", value=1_000_000, step=100000)
maturity = st.sidebar.slider("4. Maturity (Years)", 0.5, 7.0, 2.5)
turnover = st.sidebar.number_input("5. Turnover (€)", value=10_000_000)

# Exposure type as string (e.g. 'CORPORATE')
exposure_type_str = st.sidebar.selectbox("6. Exposure Type", options=[e.value for e in ExposureType])

# Validation and creation of Loan object
try:
    # Convert string to Enum for Pydantic
    exposure_type_enum = ExposureType(exposure_type_str)

    loan_data = Loan(
        id=loan_id,
        pd=default_prob,
        lgd=lgd,
        ead=ead,
        maturity=maturity,
        exposure_type=exposure_type_enum,
        turnover=turnover
    )

    # Build final JSON payload: {"loans": [{...}]}
    # model_dump() converts Pydantic object into a clean dict (including enums)
    portfolio_payload = {"loans": [loan_data.model_dump()]}

except Exception as e:
    st.error(f"Data validation error (Pydantic): {e}")
    st.stop()


# --- Scenario Selection ---
st.header("1. Stress Scenario Selection")
scenario_name = st.radio(
    "Choose the stress level applied to the portfolio:",
    options=list(SCENARIOS_INFO.keys()),
    format_func=lambda x: f"{x.upper()} : {SCENARIOS_INFO[x]}"
)

st.info(SCENARIOS_INFO[scenario_name])
st.markdown("---")

# --- Compute Button ---
if st.button("▶️ Run Stress Test and Compute Capital", type="primary"):

    with st.spinner("Calling FastAPI backend and computing capital requirements..."):

        # Call stress test endpoint
        result = call_api("assess/stress-test", portfolio_payload, scenario_name)

    if result:
        st.header("2. Prudential Assessment Results")

        # Display main metrics
        st.subheader("Impact Summary (Shock: " + scenario_name.upper() + ")")

        col1, col2, col3 = st.columns(3)

        # A. CAPITAL IMPACT
        # Determine the color (red if positive impact = capital needed)
        # Note: negative impact = capital decreases because PD > default threshold.
        impact_val = result['capital_impact']
        impact_color = "inverse" if impact_val < 0 else "normal"

        # Formatting capital impact text
        capital_impact_text = f"{impact_val:,.0f} €"

        col1.metric(
            label="Additional Required Capital",
            value=capital_impact_text,
            delta=f"Baseline: {result['baseline_metrics']['capital_requirement']:,.0f} €",
            delta_color=impact_color
        )

        st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)

        # B. METRICS DETAIL (DataFrame)

        metrics_data = {
            "Metric": ["Average PD", "Expected Loss (EL)", "Total RWA", "Capital Requirement (8%)"],
            "Baseline (No Shock)": [
                f"{result['baseline_metrics']['average_pd']:.2%}",
                f"{result['baseline_metrics']['total_expected_loss']:,.0f} €",
                f"{result['baseline_metrics']['total_rwa']:,.0f} €",
                f"{result['baseline_metrics']['capital_requirement']:,.0f} €"
            ],
            f"Stress ({scenario_name.upper()})": [
                f"{result['stressed_metrics']['average_pd']:.2%}",
                f"{result['stressed_metrics']['total_expected_loss']:,.0f} €",
                f"{result['stressed_metrics']['total_rwa']:,.0f} €",
                f"{result['stressed_metrics']['capital_requirement']:,.0f} €"
            ]
        }

        metrics_df = pd.DataFrame(metrics_data)

        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Capital Analysis")
        st.markdown(f"""
            - **Expected Loss (EL)**: EL increases significantly under stress, signalling the need for higher **accounting provisions**.
            - **RWA Calculation**: RWAs form the denominator of the solvency ratio. They are computed using the Basel Asymptotic Single Risk Factor (ASRF) model.
        """)
