import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

st.title("📊 Lean 4.0 Maturity Assessment Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filtres de sélection")

with st.sidebar.expander("Secteur de l'entreprise", expanded=True):
    sectors = st.multiselect(
        "Sélectionnez un ou plusieurs secteurs",
        options=df["Quelle est le secteur de votre entreprise ? "].unique(),
        default=df["Quelle est le secteur de votre entreprise ? "].unique(),
        help="Filtrer par secteur d'activité"
    )

with st.sidebar.expander("Taille de l'entreprise", expanded=True):
    sizes = st.multiselect(
        "Sélectionnez une ou plusieurs tailles",
        options=df["Taille entreprise "].unique(),
        default=df["Taille entreprise "].unique(),
        help="Filtrer par taille d'entreprise"
    )

with st.sidebar.expander("Niveau de maturité", expanded=True):
    maturity_levels = st.multiselect(
        "Sélectionnez un ou plusieurs niveaux",
        options=df["Maturity Level"].unique(),
        default=df["Maturity Level"].unique(),
        help="Filtrer par niveau de maturité"
    )

if st.sidebar.button("🔄 Réinitialiser les filtres"):
    # Pour réinitialiser, on recharge la page (simple workaround)
    st.experimental_rerun()

# --- Filtered Data ---
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(sectors)) &
    (df["Taille entreprise "].isin(sizes)) &
    (df["Maturity Level"].isin(maturity_levels))
]

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("📈 Moyenne Lean Score", round(filtered_df["Lean Score"].mean(), 2))
col2.metric("🖥️ Moyenne Tech Score", round(filtered_df["Tech Score"].mean(), 2))
col3.metric("🔗 Score Combiné Moyen", round(filtered_df["Combined Score"].mean(), 2))

st.markdown("---")

# --- Visualisations ---
sector_counts = filtered_df["Quelle est le secteur de votre entreprise ? "].value_counts().reset_index()
sector_counts.columns = ["Secteur", "Nombre"]

fig_sector = px.bar(
    sector_counts,
    x="Secteur",
    y="Nombre",
    title="Répartition par secteur",
    labels={"Secteur": "Secteur", "Nombre": "Nombre d'entreprises"},
    color="Nombre",
    color_continuous_scale="viridis"
)
st.plotly_chart(fig_sector, use_container_width=True)

fig_scatter = px.scatter(
    filtered_df,
    x="Lean Score",
    y="Tech Score",
    color="Maturity Level",
    title="📊 Corrélation entre Lean Score et Tech Score",
    hover_data=["Nom de l'entreprise", "Poste occupé"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

fig_hist = px.histogram(
    filtered_df,
    x="Combined Score",
    nbins=10,
    title="📏 Distribution des Scores Combinés",
    color="Maturity Level"
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("🔬 Corrélation entre les Scores")
fig, ax = plt.subplots()
sns.heatmap(filtered_df[["Lean Score", "Tech Score", "Combined Score"]].corr(), annot=True, cmap="Blues", ax=ax)
st.pyplot(fig)

fig_maturity = px.pie(
    filtered_df,
    names="Maturity Level",
    title="🏆 Répartition des niveaux de maturité"
)
st.plotly_chart(fig_maturity, use_container_width=True)

with st.expander("📄 Afficher les données brutes"):
    st.dataframe(filtered_df)
