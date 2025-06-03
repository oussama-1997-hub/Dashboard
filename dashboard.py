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
st.title("📊 Lean 4.0 Maturity Assessment Dashboard")
st.markdown("Gain insights into Lean & Digital transformation maturity across industries.")

# Sidebar Filters
with st.sidebar:
    st.header("🔎 Filters")
    sector = st.multiselect("Secteur industriel", df["Quelle est le secteur de votre entreprise ? "].unique())
    size = st.multiselect("Taille entreprise ", df["Taille entreprise "].unique())

    filtered_df = df.copy()
    if sector:
        filtered_df = filtered_df[filtered_df["Quelle est le secteur de votre entreprise ? "].isin(sector)]
    if size:
        filtered_df = filtered_df[filtered_df["Taille entreprise "].isin(size)]

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Moyenne Lean Score", f"{filtered_df['Lean Score'].mean():.2f}")
col2.metric("🤖 Moyenne Tech Score", f"{filtered_df['Tech Score'].mean():.2f}")
col3.metric("📈 Moyenne Combined Score", f"{filtered_df['Combined Score'].mean():.2f}")

st.markdown("---")

# Score Distribution
col4, col5 = st.columns(2)

with col4:
    fig = px.histogram(filtered_df, x="Lean Score", nbins=10, title="Distribution du Lean Score")
    st.plotly_chart(fig, use_container_width=True)

with col5:
    fig = px.histogram(filtered_df, x="Tech Score", nbins=10, title="Distribution du Tech Score")
    st.plotly_chart(fig, use_container_width=True)

# Maturity Levels
st.markdown("### 🏆 Répartition des niveaux de maturité")
col6, col7 = st.columns(2)

with col6:
    fig = px.pie(filtered_df, names="Maturity Level", title="Maturity Level")
    st.plotly_chart(fig, use_container_width=True)

with col7:
        # Cleaned sector column name
    secteur_col = "Quelle est le secteur de votre entreprise ? "
    
    # Count values and rename columns for clarity
    sector_counts = filtered_df[secteur_col].value_counts().reset_index()
    sector_counts.columns = ["Secteur", "Nombre"]
    
    # Bar chart
    fig = px.bar(
        sector_counts,
        x="Secteur",
        y="Nombre",
        title="Répartition par secteur",
        labels={"Secteur": "Secteur", "Nombre": "Nombre d'entreprises"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(fig, use_container_width=True)

# Scatterplot: Tech vs Lean Scores
st.markdown("### 🔬 Corrélation entre Lean et Tech Scores")
fig = px.scatter(filtered_df, x="Lean Score", y="Tech Score", color="Maturity Level",
                 hover_data=["Nom de l'entreprise", "Poste occupé", "Taille entreprise "],
                 title="Lean Score vs Tech Score par entreprise")
st.plotly_chart(fig, use_container_width=True)

# Raw Data
with st.expander("🧾 Voir les données brutes"):
    st.dataframe(filtered_df)
