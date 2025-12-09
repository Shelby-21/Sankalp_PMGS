# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Supplier Scorecard Dashboard")

@st.cache_data
def load_excel(path):
    # Reads Supplier Scorecard and Weights
    xl = pd.ExcelFile(path)
    
    # Read the main data sheet "Supplier Scorecard"
    # The file has multiple rows per supplier (Product Categories). 
    # We group by Supplier to get the average score for the scorecard.
    df = pd.read_excel(xl, "Supplier Scorecard")
    
    # Check if 'Product Category' exists and we need to aggregate
    if "Supplier" in df.columns:
        # Group by Supplier and take the mean of numeric columns
        df = df.groupby("Supplier", as_index=False).mean(numeric_only=True)
        
    # read weights
    w = pd.read_excel(xl, "Weights", usecols=[0,1], header=0)
    w.columns = ["Metric", "DefaultWeight"]
    weights = dict(zip(w["Metric"], w["DefaultWeight"]))
    return df, weights

def compute_composite(df, weights):
    # keys matching columns in the new Excel file
    keys = ["Cost Score", "Quality Score", "Delivery Score", "Sustainability Score", "Compliance Score", "Innovation Score"]
    
    # Normalize weights to sum to 1
    w = np.array([weights.get(k, 0) for k in keys], dtype=float)
    if w.sum() == 0:
        norm = np.ones_like(w) / len(w)
    else:
        norm = w / w.sum()
    
    # compute composite
    df = df.copy()
    # Ensure we only use columns that exist (in case of partial data), though we expect all keys
    valid_keys = [k for k in keys if k in df.columns]
    
    # Re-normalize if some keys are missing (defensive)
    if len(valid_keys) < len(keys):
        w = np.array([weights.get(k, 0) for k in valid_keys], dtype=float)
        norm = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)
        
    df["Composite"] = (df[valid_keys].values * norm).sum(axis=1)
    return df, dict(zip(valid_keys, norm))

def radar_plot(ax, categories, values, title=""):
    # categories: list of labels
    # values: list of values corresponding
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    # Adjusted limit to 5 based on data range (1-5)
    ax.set_ylim(0, 5)
    ax.set_title(title, pad=20)

def main():
    st.title("GenAI-Enhanced Supplier Evaluation Scorecard — Dashboard")
    st.markdown("Interactive dashboard: change weights, select supplier, compare KPIs.")

    # Left: file upload / load default
    st.sidebar.header("Data source / Controls")
    uploaded_file = st.sidebar.file_uploader("Upload corrected Excel (optional)", type=["xlsx","xls","csv"])
    
    # Define the expected columns based on the new Excel file
    expected_cols = ["Supplier", "Cost Score", "Quality Score", "Delivery Score", "Sustainability Score", "Compliance Score", "Innovation Score"]

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            # Check for columns
            if not set(expected_cols).issubset(df.columns):
                st.error(f"CSV must have columns: {', '.join(expected_cols)}")
                st.stop()
            
            # Aggregate if multiple rows per supplier
            df = df.groupby("Supplier", as_index=False).mean(numeric_only=True)
            
            # Default weights if CSV is uploaded without weights file
            weights = {
                "Cost Score": 0.25,
                "Quality Score": 0.25,
                "Delivery Score": 0.2,
                "Sustainability Score": 0.1,
                "Compliance Score": 0.1,
                "Innovation Score": 0.1
            }
        else:
            # save to a temp file because pandas read_excel can read file-like too
            df, weights = load_excel(uploaded_file)
    else:
        # default read from file in working directory
        try:
            df, weights = load_excel("Aesthetic_Supplier_Scorecard.xlsx")
        except Exception as e:
            st.error(f"Default Excel not found or error reading it: {e}")
            st.stop()

    # Show raw data (collapsed)
    with st.expander("Show raw dashboard data"):
        st.dataframe(df)

    # Read default weights from weights dict and create sliders
    st.sidebar.markdown("### Adjust Weightages (live)")
    
    # Mapping weights from Excel keys
    default_weights = {
        "Cost Score": weights.get("Cost Score", 0.25),
        "Quality Score": weights.get("Quality Score", 0.25),
        "Delivery Score": weights.get("Delivery Score", 0.20),
        "Sustainability Score": weights.get("Sustainability Score", 0.10),
        "Compliance Score": weights.get("Compliance Score", 0.10),
        "Innovation Score": weights.get("Innovation Score", 0.10)
    }

    # Sliders (Scale 0-100 for granularity, then divide by 100 for calculation)
    # We display them as percentage integers for UI, but data uses decimals (0.25)
    w_cost = st.sidebar.slider("Cost", 0, 100, int(default_weights["Cost Score"]*100), 1)
    w_quality = st.sidebar.slider("Quality", 0, 100, int(default_weights["Quality Score"]*100), 1)
    w_delivery = st.sidebar.slider("Delivery", 0, 100, int(default_weights["Delivery Score"]*100), 1)
    w_sus = st.sidebar.slider("Sustainability", 0, 100, int(default_weights["Sustainability Score"]*100), 1)
    w_comp = st.sidebar.slider("Compliance", 0, 100, int(default_weights["Compliance Score"]*100), 1)
    w_innov = st.sidebar.slider("Innovation", 0, 100, int(default_weights["Innovation Score"]*100), 1)

    weight_dict = {
        "Cost Score": w_cost/100, 
        "Quality Score": w_quality/100, 
        "Delivery Score": w_delivery/100, 
        "Sustainability Score": w_sus/100,
        "Compliance Score": w_comp/100,
        "Innovation Score": w_innov/100
    }

    total = sum([v*100 for v in weight_dict.values()])
    if total != 100:
        st.sidebar.warning(f"Weightages sum to {total}. They will be normalized internally.")

    # compute composite
    df_comp, normalized_weights = compute_composite(df, weight_dict)

    # Main layout: top KPIs and supplier select
    st.subheader("Top-line KPIs")
    col1, col2, col3 = st.columns(3)
    
    # Ensure we have data
    if not df_comp.empty:
        top_supplier = df_comp.sort_values("Composite", ascending=False).iloc[0]["Supplier"]
        col1.metric("Top Supplier (by composite)", top_supplier)
        col2.metric("Average Composite Score", f"{df_comp['Composite'].mean():.2f}")
        col3.metric("Avg Sustainability", f"{df_comp['Sustainability Score'].mean():.2f}")
    else:
        st.error("No data available.")
        st.stop()

    # Two-column: left charts, right details
    left, right = st.columns((2,1))

    # Left: Comparison bar + radar for selected supplier
    with left:
        st.subheader("Cost Score comparison")
        # Sorting by Cost Score for the bar chart
        sort_col = "Cost Score"
        if sort_col in df_comp.columns:
            chart_df = df_comp[["Supplier", sort_col]].sort_values(sort_col, ascending=False)
            fig1, ax1 = plt.subplots(figsize=(8,3))
            ax1.bar(chart_df["Supplier"], chart_df[sort_col])
            ax1.set_ylim(0, 5) # Adjusted for 1-5 scale
            ax1.set_ylabel("Cost Score")
            plt.xticks(rotation=45)
            st.pyplot(fig1)

        st.markdown("---")
        st.subheader("Supplier Radar Chart")
        supplier_list = df_comp["Supplier"].tolist()
        selected = st.selectbox("Choose supplier:", supplier_list, index=0)

        # radar chart
        categories = ["Cost Score", "Quality Score", "Delivery Score", "Sustainability Score", "Compliance Score", "Innovation Score"]
        sel_row = df_comp[df_comp["Supplier"] == selected].iloc[0]
        values = [sel_row[c] for c in categories]
        
        fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        radar_plot(ax2, categories, values, title=f"{selected} Profile")
        st.pyplot(fig2)

    # Right: table and weight breakdown
    with right:
        st.subheader("Supplier Scores (Ranked)")
        st.dataframe(df_comp.sort_values("Composite", ascending=False).reset_index(drop=True))

        st.markdown("### Weight normalization (applied):")
        for k,v in normalized_weights.items():
            st.write(f"{k}: {v:.2f}")

        st.markdown("---")
        st.markdown("### Export")
        if st.button("Download current scores as CSV"):
            tmp = df_comp.to_csv(index=False)
            st.download_button("Download CSV", data=tmp, file_name="supplier_scores.csv", mime="text/csv")

    st.markdown("### Notes")
    st.markdown("- Data loaded from 'Supplier Scorecard' sheet.")
    st.markdown("- Scores are averaged by Supplier (aggregating across Product Categories).")
    st.markdown("- Use sliders to adjust weightings.")

if __name__ == "__main__":
    main()
