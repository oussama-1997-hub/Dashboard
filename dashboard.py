import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit config must be first
st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

# 📥 Load data
@st.cache_data
def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

st.title("📊 Lean 4.0 Maturity Assessment Dashboard")

# Sidebar filters
st.sidebar.header("🔎 Filters")
sectors = st.sidebar.multiselect("Secteur", df["Quelle est le secteur de votre entreprise ? "].unique(), default=df["Quelle est le secteur de votre entreprise ? "].unique())
sizes = st.sidebar.multiselect("Taille d’entreprise", df["Taille entreprise"].unique(), default=df["Taille entreprise"].unique())
maturity_levels = st.sidebar.multiselect("Niveau de maturité", df["Maturity Level"].unique(), default=df["Maturity Level"].unique())

# Filtered data
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille entreprise"].isin(sizes)) &
    (df["Maturity Level"].isin(maturity_levels))
]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("📈 Moyenne Lean Score", round(filtered_df["Lean Score"].mean(), 2))
col2.metric("🖥️ Moyenne Tech Score", round(filtered_df["Tech Score"].mean(), 2))
col3.metric("🔗 Score Combiné Moyen", round(filtered_df["Combined Score"].mean(), 2))

st.markdown("---")

# Sector Distribution
fig_sector = px.bar(
    filtered_df["Quelle est le secteur de votre entreprise ? "].value_counts().reset_index(),
    x="index",
    y="Quelle est le secteur de votre entreprise ? ",
    title="🔧 Répartition des entreprises par secteur",
    labels={"index": "Secteur", "Quelle est le secteur de votre entreprise ? ": "Nombre"}
)
st.plotly_chart(fig_sector, use_container_width=True)

# Lean vs Tech Score Scatter
fig_scatter = px.scatter(
    filtered_df,
    x="Lean Score",
    y="Tech Score",
    color="Maturity Level",
    title="📊 Corrélation entre Lean Score et Tech Score",
    hover_data=["Nom de l'entreprise", "Poste occupé"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Score distribution
fig_hist = px.histogram(
    filtered_df,
    x="Combined Score",
    nbins=10,
    title="📏 Distribution des Scores Combinés",
    color="Maturity Level"
)
st.plotly_chart(fig_hist, use_container_width=True)

# Heatmap: Correlation between scores
st.subheader("🔬 Corrélation entre les Scores")
fig, ax = plt.subplots()
sns.heatmap(filtered_df[["Lean Score", "Tech Score", "Combined Score"]].corr(), annot=True, cmap="Blues", ax=ax)
st.pyplot(fig)

# Maturity Level Distribution
fig_maturity = px.pie(
    filtered_df,
    names="Maturity Level",
    title="🏆 Répartition des niveaux de maturité"
)
st.plotly_chart(fig_maturity, use_container_width=True)

# Optional: Raw data
with st.expander("📄 Afficher les données brutes"):
    st.dataframe(filtered_df)

