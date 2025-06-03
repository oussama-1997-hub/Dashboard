import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_excel("processed_data.xlsx")

df = load_data()

st.title("📊 Lean 4.0 Maturity Assessment Dashboard")

st.sidebar.header("🔎 Filtres")

# Secteur (multiselect classique)
sectors = st.sidebar.multiselect(
    label="Secteur",
    options=sorted(df["Quelle est le secteur de votre entreprise ? "].dropna().unique()),
    default=sorted(df["Quelle est le secteur de votre entreprise ? "].dropna().unique()),
    help="Filtrer par secteur"
)

# Taille entreprise (slider numérique)
taille_min = int(df["Taille entreprise "].min())
taille_max = int(df["Taille entreprise "].max())
taille_range = st.sidebar.slider(
    label="Taille d’entreprise",
    min_value=taille_min,
    max_value=taille_max,
    value=(taille_min, taille_max),
    step=1,
    help="Filtrer par taille d’entreprise (nombre d’employés)"
)

# Niveau de maturité (slider numérique)
maturite_min = int(df["Maturity Level"].min())
maturite_max = int(df["Maturity Level"].max())
maturite_range = st.sidebar.slider(
    label="Niveau de maturité",
    min_value=maturite_min,
    max_value=maturite_max,
    value=(maturite_min, maturite_max),
    step=1,
    help="Filtrer par niveau de maturité"
)

# Bouton reset filtres
if st.sidebar.button("🔄 Réinitialiser les filtres"):
    st.experimental_rerun()

# Filtrage
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille entreprise "].between(taille_range[0], taille_range[1])) &
    (df["Maturity Level"].between(maturite_range[0], maturite_range[1]))
]

# KPIs
st.markdown("### 📈 Indicateurs clés")
col1, col2, col3 = st.columns(3)
col1.metric("Lean Score moyen", f"{filtered_df['Lean Score'].mean():.2f}")
col2.metric("Tech Score moyen", f"{filtered_df['Tech Score'].mean():.2f}")
col3.metric("Score combiné moyen", f"{filtered_df['Combined Score'].mean():.2f}")

st.markdown("---")

# Visualisations
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

st.markdown("### Distribution des Scores Combinés")
fig_hist = px.histogram(
    filtered_df,
    x="Combined Score",
    nbins=10,
    color="Maturity Level",
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)

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

st.markdown("### Répartition des niveaux de maturité")
fig_maturity = px.pie(
    filtered_df,
    names="Maturity Level",
    height=400
)
st.plotly_chart(fig_maturity, use_container_width=True)

with st.expander("📄 Afficher les données brutes filtrées"):
    st.dataframe(filtered_df.reset_index(drop=True))
