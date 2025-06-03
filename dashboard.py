import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ Set page config at the top (MUST BE FIRST Streamlit command)
st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

# 📥 Load data
@st.cache_data
def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

# ✅ Sidebar filters
st.sidebar.title("🔍 Filtres")
secteurs = df["Quelle est le secteur de votre entreprise ? "].dropna().unique()
secteur_selectionne = st.sidebar.multiselect("Secteurs d'activité :", secteurs, default=secteurs)

# ✅ Filtered dataframe
filtered_df = df[df["Quelle est le secteur de votre entreprise ? "].isin(secteur_selectionne)]

# ✅ Main Title
st.title("📊 Tableau de Bord de Maturité Lean 4.0")

# ✅ Clean sector names (basic normalization)
filtered_df["secteur_clean"] = (
    filtered_df["Quelle est le secteur de votre entreprise ? "]
    .str.strip()
    .str.lower()
    .str.replace("industrie ", "")
    .str.replace("industries ", "")
    .str.replace("aéronautique", "aéronautique")
    .str.replace("automobile", "automobile")
)

# ✅ KPI cards
col1, col2, col3 = st.columns(3)
col1.metric("Nombre de réponses", len(filtered_df))
col2.metric("Secteurs uniques", filtered_df["secteur_clean"].nunique())
col3.metric("Taux de complétion", f"{filtered_df.notnull().mean().mean()*100:.1f}%")

# ✅ Bar chart: Répartition par secteur
st.subheader("📌 Répartition des Réponses par Secteur")
secteur_count = filtered_df["secteur_clean"].value_counts().reset_index()
secteur_count.columns = ["Secteur", "Nombre"]

fig = px.bar(
    secteur_count,
    x="Secteur",
    y="Nombre",
    title="Répartition par secteur",
    labels={"Nombre": "Nombre de réponses"},
    color="Secteur"
)
st.plotly_chart(fig, use_container_width=True, key="bar_chart_secteurs")

# ✅ Radar Chart: Moyenne des dimensions
dimensions = [
    "Leadership - Engagement Lean ",
    "Leadership - Engagement DT",
    "Leadership - Stratégie ",
    "Leadership - Communication",
    "Supply Chain - Collaboration inter-organisationnelle",
    "Supply Chain - Traçabilité",
    "Supply Chain - Impact sur les employées",
    "Opérations - Standardisation des processus",
    "Opérations - Juste-à-temps (JAT)",
    "Opérations - Gestion des résistances",
    "Technologies - Connectivité et gestion des données",
    "Technologies - Automatisation",
    "Technologies - Pilotage du changement",
    "Organisation apprenante  - Formation et développement des compétences",
    "Organisation apprenante  - Collaboration et Partage des Connaissances",
    "Organisation apprenante  - Flexibilité organisationnelle"
]

# Remove empty or non-float columns
valid_dimensions = [col for col in dimensions if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])]

radar_data = pd.DataFrame({
    "Dimension": valid_dimensions,
    "Score Moyen": [filtered_df[col].mean() for col in valid_dimensions]
})

fig_radar = px.line_polar(radar_data, r='Score Moyen', theta='Dimension', line_close=True,
                          title="Maturité Moyenne par Dimension", markers=True)
fig_radar.update_traces(fill='toself')
st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart_maturite")

# ✅ Show filtered data
with st.expander("📄 Voir les données filtrées"):
    st.dataframe(filtered_df)

# ✅ Footer
st.markdown("---")
st.markdown("Réalisé avec ❤️ par Oussama Ben Ali")
