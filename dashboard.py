import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Config Streamlit
st.set_page_config(page_title="Lean 4.0 Maturity Dashboard", layout="wide")

# Charger les données
@st.cache_data

def load_data():
    df = pd.read_excel("processed_data.xlsx")
    return df

df = load_data()

st.title("📊 Tableau de bord : Analyse de la maturité Lean 4.0")

# =============================
# 🔧 Prétraitement
# =============================

# Encodage Taille entreprise en code ordinal pour slider
size_mapping = {"Petite": "< 50", "Moyenne": "[50 , 199]", "Grande": "[200 , 499]", "Très grande": ">= 500"}
df["Taille_code"] = df["Taille entreprise "].map(size_mapping)

# =============================
# 🎚️ Filtres
# =============================
st.sidebar.header("🎛️ Filtres")

secteurs = st.sidebar.multiselect(
    "Secteur",
    options=df["Quelle est le secteur de votre entreprise ? "].unique(),
    default=df["Quelle est le secteur de votre entreprise ? "].unique()
)

taille_min, taille_max = 1, 4
taille_range = st.sidebar.slider(
    "Taille d'entreprise (1: Petite - 4: Très grande)",
    min_value=1,
    max_value=4,
    value=(1, 4)
)

maturity_range = st.sidebar.slider(
    "Niveau de maturité globale Lean 4.0",
    min_value=int(df["maturity_level_int"].min()),
    max_value=int(df["maturity_level_int"].max()),
    value=(int(df["maturity_level_int"].min()), int(df["maturity_level_int"].max()))
)

# Application des filtres
filtered_df = df[
    (df["Quelle est le secteur de votre entreprise ? "].isin(secteurs)) &
    (df["Taille_code"].between(taille_range[0], taille_range[1])) &
    (df["maturity_level_int"].between(maturity_range[0], maturity_range[1]))
]

# =============================
# 📌 Indicateurs Clés
# =============================
col1, col2, col3 = st.columns(3)
col1.metric("📈 Moyenne Maturité Globale", round(filtered_df["maturity_level_int"].mean(), 2))
col2.metric("🧠 Maturité Lean", round(filtered_df["Lean_level_int"].mean(), 2))
col3.metric("💻 Maturité Industrie 4.0", round(filtered_df["Digital_level_int"].mean(), 2))

st.markdown("---")

# =============================
# 📊 Visualisations
# =============================

# Répartition par secteur
st.subheader("🔍 Répartition des entreprises par secteur")
fig_secteur = px.histogram(filtered_df, x="Quelle est le secteur de votre entreprise ? ", color_discrete_sequence=['indigo'])
st.plotly_chart(fig_secteur, use_container_width=True)

# Scatter Lean vs Digital
st.subheader("💡 Corrélation entre Maturité Lean et Industrie 4.0")
fig_scatter = px.scatter(
    filtered_df,
    x="Lean_level_int",
    y="Digital_level_int",
    color="maturity_level_int",
    hover_data=["Taille entreprise ", "Quelle est le secteur de votre entreprise ? "]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Heatmap corrélation
st.subheader("📈 Corrélation entre les indicateurs de maturité")
fig, ax = plt.subplots()
sns.heatmap(filtered_df[["maturity_level_int", "Lean_level_int", "Digital_level_int"]].corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# =============================
# 🔎 EDA : Distribution des scores
# =============================
st.subheader("📏 Distribution des niveaux de maturité")
fig_hist = px.histogram(filtered_df, x="maturity_level_int", nbins=10, color_discrete_sequence=["darkcyan"])
st.plotly_chart(fig_hist, use_container_width=True)



# =============================
# 🌲 Arbre de décision
# =============================
st.subheader("🤖 Prédiction de la maturité technologique")
features = [col for col in df.columns if any(dim in col for dim in ["Leadership", "Supply Chain", "Opérations", "Technologies", "Organisation apprenante"])]
X = filtered_df[features].fillna(0)
y = filtered_df["Digital_level_int"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

st.markdown("**Arbre de Décision (max_depth=4)**")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=True, filled=True, ax=ax)
st.pyplot(fig)

st.markdown("**Rapport de performance :**")
preds = clf.predict(X_test)
st.text(classification_report(y_test, preds))

# =============================
# 📄 Données brutes
# =============================
with st.expander("📋 Afficher les données brutes"):
    st.dataframe(filtered_df)
