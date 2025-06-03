import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

# Chargement des données avec cache
@st.cache_data
def load_data():
    return pd.read_excel("processed_data.xlsx")

df = load_data()

# Titre principal
st.title("📊 Lean 4.0 Maturity Assessment Dashboard")

# --- Sidebar Filtres compactes ---
st.sidebar.header("🔎 Filtres")

# Organiser les filtres en colonnes pour gagner de la place
col1, col2, col3 = st.sidebar.columns([1,1,1])

with col1:
    sectors = st.multiselect(
        label="Secteur",
        options=sorted(df["Quelle est le secteur de votre entreprise ? "].dropna().unique()),
        default=sorted(df["Quelle est le secteur de votre entreprise ? "].dropna().unique()),
        help="Filtrer par secteur"
    )

with col2:
    sizes = st.multiselect(
        label="Taille",
        options=sorted(df["Taille entreprise "].dropna().unique()),
        default=sorted(df["Taille entreprise "].dropna().unique()),
        help="Filtrer par taille d'entreprise"
    )

with col3:
    maturity_levels = st.multiselect(
        label="Maturité",
        options=sorted(df["Maturity Level"].dropna().unique()),
        default=sorted(df["Maturity Level"].dropna().unique()),
        help="Filtrer par niveau de maturité"
    )

if st.sidebar.button("🔄 Réinitialiser les filtres"):
    st.experimental_rerun()

# --- Filtrage des données ---
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille entreprise "].isin(sizes)) &
    (df["Maturity Level"].isin(maturity_levels))
]

# --- KPI ---
st.markdown("### 📈 Indicateurs clés")
col1, col2, col3 = st.columns(3)
col1.metric("Lean Score moyen", f"{filtered_df['Lean Score'].mean():.2f}")
col2.metric("Tech Score moyen", f"{filtered_df['Tech Score'].mean():.2f}")
col3.metric("Score combiné moyen", f"{filtered_df['Combined Score'].mean():.2f}")

st.markdown("---")

# --- Visualisations ---

# Répartition par secteur
st.markdown("### Répartition par secteur")
sector_counts = filtered_df["Quelle est le secteur de votre entreprise ? "].value_counts().reset_index()
sector_counts.columns = ["Secteur", "Nombre"]

fig_sector = px.bar(
    sector_counts,
    x="Secteur",
    y="Nombre",
    color="Nombre",
    color_continuous_scale="Viridis",
    labels={"Secteur": "Secteur", "Nombre": "Nombre d'entreprises"},
    height=400
)
st.plotly_chart(fig_sector, use_container_width=True)

# Scatter Lean vs Tech Score
st.markdown("### Corrélation Lean Score vs Tech Score")
fig_scatter = px.scatter(
    filtered_df,
    x="Lean Score",
    y="Tech Score",
    color="Maturity Level",
    hover_data=["Nom de l'entreprise", "Poste occupé"],
    height=400
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Distribution des scores combinés
st.markdown("### Distribution des Scores Combinés")
fig_hist = px.histogram(
    filtered_df,
    x="Combined Score",
    nbins=10,
    color="Maturity Level",
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)

# Heatmap corrélation scores
st.markdown("### Corrélation entre les scores")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(
    filtered_df[["Lean Score", "Tech Score", "Combined Score"]].corr(),
    annot=True,
    cmap="Blues",
    ax=ax,
    fmt=".2f"
)
st.pyplot(fig)

# Répartition des niveaux de maturité (camembert)
st.markdown("### Répartition des niveaux de maturité")
fig_maturity = px.pie(
    filtered_df,
    names="Maturity Level",
    height=400
)
st.plotly_chart(fig_maturity, use_container_width=True)

# --- Affichage des données brutes ---
with st.expander("📄 Afficher les données brutes filtrées"):
    st.dataframe(filtered_df.reset_index(drop=True))

