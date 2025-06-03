import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Lean 4.0 Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("processed_data.xlsx")

df = load_data()

# Title and Description
st.title("📊 Lean 4.0 & Digital Transformation Dashboard")
st.markdown("""
Explore company maturity across Lean 4.0 dimensions such as leadership, operations, technology, and learning organization.
""")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔍 Total Rows", f"{df.shape[0]}")
col2.metric("🏭 Unique Sectors", df["secteur industrie"].nunique())
col3.metric("🌍 Countries", df["Pays"].nunique() if "Pays" in df else "N/A")
col4.metric("📂 Columns", f"{df.shape[1]}")

st.markdown("---")

# Sector Distribution
st.subheader("📌 Industry Sector Distribution")
fig1 = px.histogram(df, x="secteur industrie", color="secteur industrie", title="Distribution des secteurs industriels")
fig1.update_layout(xaxis_title="Secteur", yaxis_title="Count", showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# Leadership Radar Chart
st.subheader("👔 Leadership Indicators")
leadership_cols = [
    "Leadership - Engagement Lean ",
    "Leadership - Engagement DT",
    "Leadership - Stratégie ",
    "Leadership - Communication"
]
if all(col in df.columns for col in leadership_cols):
    mean_values = df[leadership_cols].mean()
    radar_df = pd.DataFrame({
        "Dimension": leadership_cols,
        "Score": mean_values.values
    })

    fig2 = px.line_polar(radar_df, r='Score', theta='Dimension', line_close=True,
                         title="Leadership Radar Chart", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# Heatmap of correlation
st.subheader("🔥 Correlation Heatmap")
corr_cols = df.select_dtypes(include=['float64', 'int64'])
if not corr_cols.empty:
    corr_matrix = corr_cols.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig3, use_container_width=True)

# Data table
st.subheader("📋 Raw Data Preview")
st.dataframe(df, use_container_width=True)
