import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Supplier Scorecard Dashboard",
    layout="wide"
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    file_path = "Aesthetic_Supplier_Scorecard.xlsx"

    scorecard = pd.read_excel(file_path, sheet_name="Supplier Scorecard")
    weights = pd.read_excel(file_path, sheet_name="Weights")
    dashboard = pd.read_excel(file_path, sheet_name="Dashboard Prep")

    return scorecard, weights, dashboard

scorecard, weights, dashboard = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("Supplier Evaluation Scorecard Dashboard")
st.markdown("This dashboard visualizes supplier performance using the weighted scorecard.")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

selected_supplier = st.sidebar.selectbox(
    "Select Supplier",
    options=["All"] + dashboard["Supplier Name"].tolist()
)

metric_to_view = st.sidebar.selectbox(
    "Select KPI to Visualize",
    options=["Composite Score", "Technical", "Quality", "Financial", "ESG", "Innovation"]
)

# -----------------------------
# Data Filtering
# -----------------------------
if selected_supplier != "All":
    filtered_df = dashboard[dashboard["Supplier Name"] == selected_supplier]
else:
    filtered_df = dashboard.copy()

# -----------------------------
# Section 1: Weights Table
# -----------------------------
st.subheader("Current Weight Distribution")
st.dataframe(weights, use_container_width=True)

# -----------------------------
# Section 2: Composite Score Bar Chart
# -----------------------------
st.subheader(f"{metric_to_view} Comparison")

fig = px.bar(
    filtered_df,
    x="Supplier Name",
    y=metric_to_view,
    title=f"{metric_to_view} by Supplier",
    text_auto=True
)
fig.update_layout(xaxis_title="", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Section 3: Supplier Radar Chart
# -----------------------------
st.subheader("Radar Chart (Performance Across KPIs)")

def radar_chart(row):
    df = pd.DataFrame({
        "KPI": ["Technical", "Quality", "Financial", "ESG", "Innovation"],
        "Score": [
            row["Technical"], row["Quality"], row["Financial"],
            row["ESG"], row["Innovation"]
        ]
    })
    fig_radar = px.line_polar(
        df,
        r="Score",
        theta="KPI",
        line_close=True,
        title=f"{row['Supplier Name']} - KPI Radar Chart",
        range_r=[0, 100]
    )
    return fig_radar

if selected_supplier == "All":
    st.info("Select a single supplier to view their radar chart.")
else:
    row = filtered_df.iloc[0]
    st.plotly_chart(radar_chart(row), use_container_width=True)

# -----------------------------
# Section 4: Detailed Supplier Table
# -----------------------------
st.subheader("Supplier Detailed Performance Table")
st.dataframe(scorecard, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Dashboard generated using Streamlit | Supplier Scorecard System for Procurement")
